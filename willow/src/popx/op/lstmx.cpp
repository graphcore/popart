// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lstmx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

LSTMOpx::LSTMOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LSTMOp>(op, {Onnx::Operators::LSTM_1, Onnx::Operators::LSTM_7});
}

// Only create an intermediate tensor if it is consumed or used as a anchor
std::unique_ptr<poplar::Tensor> LSTMOpx::createIntermediate() const {
  if (getOp<LSTMOp>().isTraining()) {
    return std::make_unique<poplar::Tensor>();
  } else {
    return std::unique_ptr<poplar::Tensor>(nullptr);
  }
}

void LSTMOpx::grow(poplar::program::Sequence &prog) const {
  prepareWeights(prog);
  growBias(prog);

  auto init_state = getInitialState();
  prepareInitialState(init_state, prog);

  auto intermediate = createIntermediate();
  poplar::Tensor output, cell_state;
  auto input                   = getInput(prog);
  std::tie(output, cell_state) = popnn::lstm::lstmFwd(graph(),
                                                      createLSTMParams(),
                                                      init_state,
                                                      input,
                                                      *weights,
                                                      intermediate.get(),
                                                      prog,
                                                      debugPrefix("lstmFwd"),
                                                      dv_p->lstmOptions,
                                                      &dv_p->matmulCache);

  if (intermediate) {
    setOutTensor(LSTMOp::getIntermediatesPassThroughIndex(), *intermediate);
  }

  reshapeAndInsert(LSTMOp::getOutputOutIndex(), output);

  auto output_h_state = output[createLSTMParams().timeSteps - 1];

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  reshapeAndInsert(LSTMOp::getHiddenStateOutIndex(),
                   cloneNcopy(prog, output_h_state));
  reshapeAndInsert(LSTMOp::getCellStateOutIndex(), cell_state);

  setOutTensor(LSTMOp::getInitStateOutputPassThroughIndex(), init_state.output);
  setOutTensor(LSTMOp::getInitStateCellStatePassThroughIndex(),
               init_state.cellState);
  setOutTensor(LSTMOp::getInputWeightsPassThroughIndex(),
               weights->inputWeights);
  setOutTensor(LSTMOp::getOutputWeightsPassThroughIndex(),
               weights->outputWeights);
  setOutTensor(LSTMOp::getBiasesPassThroughIndex(), weights->biases);

  // TODO T18126 register this alias or insert a cloneNcopy
  setOutTensor(LSTMOp::getInputPassThroughIndex(), input);

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  setOutTensor(LSTMOp::getOutputPassThroughIndex(), cloneNcopy(prog, output));
}

void LSTMOpx::reshapeAndInsert(OutIndex index,
                               const poplar::Tensor &tensor) const {
  if (getOp<LSTMOp>().hasOutput(index)) {
    setOutTensor(index, tensor.reshape(outInfo(index).shape_szt()));
  }
}

void LSTMOpx::growBias(poplar::program::Sequence &prog) const {
  // bias in onnx is shape [num_directions, 8 * hidden_size]
  // bias in poplibs is [4, hidden_size]
  auto &lstm_op        = getOp<LSTMOp>();
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(getLSTMWeights().biases, false);

  if (lstm_op.hasBiasInput()) {
    auto bias_input = getInTensor(LSTMOp::getBiasInIndex());

    poplar::program::Copy copyProg(bias_input.slice(0, 4 * hidden_size, 1),
                                   biases);
    prog.add(copyProg);

    popops::mapInPlace(graph(),
                       popops::expr::BinaryOpType::ADD,
                       biases,
                       bias_input.slice(4 * hidden_size, 8 * hidden_size, 1),
                       prog,
                       debugPrefix("add"));
  } else {
    popops::zero(graph(), biases, prog, debugPrefix("zero"));
  }
}

InputCreatorType LSTMOpx::getInputCreatorType(InIndex index) const {
  if (index == LSTMOp::getInputInIndex() ||
      index == LSTMOp::getWeightsInIndex() ||
      index == LSTMOp::getRecurrenceInIndex() ||
      index == LSTMOp::getInitialHInIndex() ||
      index == LSTMOp::getInitialCInIndex()) {
    return InputCreatorType::CANCREATE;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor LSTMOpx::createInput(InIndex index, const std::string &) const {
  createdInputs.insert(index);

  if (index == LSTMOp::getInputInIndex()) {
    return createLSTMInput();
  } else if (index == LSTMOp::getWeightsInIndex()) {
    auto inputWeights = getLSTMWeights().inputWeights;
    return reshapePoplibWeightsForOnnx(inputWeights, true);
  } else if (index == LSTMOp::getRecurrenceInIndex()) {
    auto outputWeights = getLSTMWeights().outputWeights;
    return reshapePoplibWeightsForOnnx(outputWeights, true);
  } else if (index == LSTMOp::getInitialCInIndex()) {
    auto &lstm_op = getOp<LSTMOp>();

    unsigned batch_size     = static_cast<unsigned>(lstm_op.getBatchSize());
    unsigned hidden_size    = static_cast<unsigned>(lstm_op.getHiddenSize());
    unsigned num_directions = static_cast<unsigned>(lstm_op.getNumDirections());

    auto init_c = getInitialState().cellState;
    return init_c.reshape({num_directions, batch_size, hidden_size});
  } else if (index == LSTMOp::getInitialHInIndex()) {
    auto &lstm_op = getOp<LSTMOp>();

    unsigned batch_size     = static_cast<unsigned>(lstm_op.getBatchSize());
    unsigned hidden_size    = static_cast<unsigned>(lstm_op.getHiddenSize());
    unsigned num_directions = static_cast<unsigned>(lstm_op.getNumDirections());

    auto init_h = getInitialState().output;
    return init_h.reshape({num_directions, batch_size, hidden_size});
  } else {
    throw error("LSTMOpx::createInput is not supported for index {}", index);
  }
}

bool LSTMOpx::inputCreated(InIndex index) const {
  return createdInputs.count(index) > 0;
}

poplar::Tensor
LSTMOpx::reshapePoplibWeightsForOnnx(poplar::Tensor poplib_weights,
                                     bool transpose) {
  // ONNX expects input weights in shape [num_directions, 4*hidden_size, K]
  // where
  //   num_directions is always 1 for popart
  //   and K is either input_size or hidden_size, for the inputWeights or
  //   outputWeights respectivly
  // and order is W[iofc]
  //
  // poplibs expects weights in shape [4, K, hidden_size]
  // and order is W[fico]
  std::vector<poplar::Interval> intervals{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
  auto slices = poplib_weights.slices(intervals, 0);

  if (transpose) {
    for (int i = 0; i < slices.size(); i++) {
      slices[i] = slices[i].dimShuffle({0, 2, 1});
    }
  }

  auto wf = slices[0];
  auto wi = slices[1];
  auto wc = slices[2];
  auto wo = slices[3];

  return poplar::concat({wi, wo, wf, wc}, 1);
}

poplar::Tensor LSTMOpx::createLSTMInput() const {
  auto lstm_params = createLSTMParams();
  auto options     = dv_p->lstmOptions;
  auto cache       = &dv_p->matmulCache;

  return popnn::lstm::createInput(
      graph(), lstm_params, debugPrefix("input"), options, cache);
}

popnn::lstm::LstmState LSTMOpx::getInitialState() const {
  if (!initial_state) {
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;
    auto lstm_params = createLSTMParams();

    initial_state = createInitialState(
        graph(), lstm_params, debugPrefix("initialState"), options, cache);
  }

  return *initial_state;
}

popnn::lstm::LstmWeights LSTMOpx::getLSTMWeights() const {
  if (!weights) {
    auto lstm_params = createLSTMParams();
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;

    weights = createWeights(
        graph(), lstm_params, debugPrefix("weights"), options, cache);
  }

  return *weights;
}

popnn::lstm::LstmParams LSTMOpx::createLSTMParams(const LSTMOp &lstm_op) {
  auto in_info         = lstm_op.inInfo(LSTMOp::getInputInIndex());
  auto seq_length      = static_cast<unsigned>(lstm_op.getSeqLength());
  auto batch_size      = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned input_size  = static_cast<unsigned>(lstm_op.getInputSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto params = popnn::lstm::LstmParams(
      popType(in_info), batch_size, seq_length, {input_size, hidden_size});
  return params;
}

popnn::lstm::LstmParams LSTMOpx::createLSTMParams() const {
  auto &lstm_op = getOp<LSTMOp>();
  return createLSTMParams(lstm_op);
}

std::vector<TensorId> LSTMOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

void LSTMOpx::prepareWeights(poplar::program::Sequence &prog) const {
  // check to see if the weights were created
  if (!inputCreated(LSTMOp::getWeightsInIndex())) {
    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getWeightsInIndex()),
        reshapePoplibWeightsForOnnx(getLSTMWeights().inputWeights, true)));
  }
  if (!inputCreated(LSTMOp::getRecurrenceInIndex())) {
    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getRecurrenceInIndex()),
        reshapePoplibWeightsForOnnx(getLSTMWeights().outputWeights, true)));
  }
}

poplar::Tensor LSTMOpx::getInput(poplar::program::Sequence &prog) const {
  if (!inputCreated(LSTMOp::getInputInIndex())) {
    auto input     = createInput(LSTMOp::getInputInIndex(), "input");
    auto raw_input = getInTensor(LSTMOp::getInputInIndex());
    prog.add(poplar::program::Copy(raw_input, input));
    return input;
  } else {
    return getInTensor(LSTMOp::getInputInIndex());
  }
}

void LSTMOpx::prepareInitialState(popnn::lstm::LstmState &init_state,
                                  poplar::program::Sequence &prog) const {
  auto &lstm_op = getOp<LSTMOp>();
  auto hasInitC = lstm_op.hasInitialCInput();
  auto hasInitH = lstm_op.hasInitialHInput();

  if (!hasInitC && !hasInitH) {
    zeroInitialState(graph(), init_state, prog, debugPrefix());
  } else if (!hasInitC) {
    popops::zero(graph(), init_state.cellState, prog, debugPrefix());
  } else if (!hasInitH) {
    popops::zero(graph(), init_state.output, prog, debugPrefix());
  }

  // Check the inputs have been created
  if (hasInitC && !inputCreated(LSTMOp::getInitialCInIndex())) {
    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getInitialCInIndex()),
        createInput(LSTMOp::getInitialCInIndex(), debugPrefix("initC"))));
  }
  if (hasInitH && !inputCreated(LSTMOp::getInitialHInIndex())) {
    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getInitialHInIndex()),
        createInput(LSTMOp::getInitialHInIndex(), debugPrefix("initH"))));
  }
}

LSTMGradOpx::LSTMGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LSTMGradOp>(op, Onnx::GradOperators::LSTMGrad);
}

void LSTMGradOpx::grow(poplar::program::Sequence &prog) const {
  popnn::lstm::LstmState init_state;
  init_state.output = getInTensor(LSTMGradOp::getInitStateOutputInIndex());
  init_state.cellState =
      getInTensor(LSTMGradOp::getInitStateCellStateInIndex());

  popnn::lstm::LstmWeights weights;
  weights.inputWeights  = getInTensor(LSTMGradOp::getInputWeightsInIndex());
  weights.outputWeights = getInTensor(LSTMGradOp::getOutputWeightsInIndex());
  weights.biases        = getInTensor(LSTMGradOp::getBiasesInIndex());

  auto intermediates  = getInTensor(LSTMGradOp::getIntermediatesInIndex());
  auto forward_input  = getInTensor(LSTMGradOp::getInputInIndex());
  auto forward_output = getInTensor(LSTMGradOp::getOutputInIndex());

  auto &lstm_grad_op  = getOp<LSTMGradOp>();
  auto &lstm_op       = lstm_grad_op.getForwardOp();
  auto batch_size     = static_cast<unsigned>(lstm_op.getBatchSize());
  auto hidden_size    = static_cast<unsigned>(lstm_op.getHiddenSize());
  auto seq_length     = static_cast<unsigned>(lstm_op.getSeqLength());
  auto num_directions = static_cast<unsigned>(lstm_op.getNumDirections());
  auto lstm_params    = createLSTMParams();

  auto output_grad = getInTensor(LSTMGradOp::getOutputGradInIndex())
                         .reshape({seq_length, batch_size, hidden_size});

  auto output_c_grad = getCellStateGrad();
  auto output_h_grad = getHiddenStateGrad();

  // TODO find out what this is for
  // it's done in tensorflow and enigma
  auto output_grad_copy = cloneNcopy(prog, output_grad);
  popops::addInPlace(graph(),
                     output_grad_copy[output_grad_copy.dim(0) - 1],
                     output_h_grad,
                     prog,
                     debugPrefix());

  poplar::Tensor input_grad;
  popnn::lstm::LstmWeights weights_grad;

  auto init_state_grad = lstmBwdWithWU(graph(),
                                       lstm_params,
                                       prog,
                                       init_state,
                                       intermediates,
                                       weights,
                                       forward_input,
                                       forward_output,
                                       output_grad_copy,
                                       &output_c_grad,
                                       &input_grad,
                                       weights_grad,
                                       debugPrefix("lstmBwdWithWU"),
                                       dv_p->lstmOptions,
                                       &dv_p->matmulCache);

  setOutTensor(LSTMGradOp::getInputOutIndex(), input_grad);
  setOutTensor(
      LSTMGradOp::getWeightsOutIndex(),
      LSTMOpx::reshapePoplibWeightsForOnnx(weights_grad.inputWeights, true));
  setOutTensor(
      LSTMGradOp::getRecurrenceOutIndex(),
      LSTMOpx::reshapePoplibWeightsForOnnx(weights_grad.outputWeights, true));

  if (lstm_op.hasBiasInput()) {
    auto b_grad =
        LSTMOpx::reshapePoplibWeightsForOnnx(weights_grad.biases, false);
    setOutTensor(LSTMGradOp::getBiasOutIndex(),
                 poplar::concat({b_grad, b_grad}, 1));
  }
  if (lstm_op.hasInitialHInput()) {
    auto init_h = init_state_grad.output;
    setOutTensor(LSTMGradOp::getInitialHOutIndex(),
                 init_h.reshape({num_directions, batch_size, hidden_size}));
  }
  if (lstm_op.hasInitialCInput()) {
    auto init_c = init_state_grad.cellState;
    setOutTensor(LSTMGradOp::getInitialCOutIndex(),
                 init_c.reshape({num_directions, batch_size, hidden_size}));
  }
}

poplar::Tensor LSTMGradOpx::getCellStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();
  auto &lstm_op      = lstm_grad_op.getForwardOp();

  unsigned batch_size  = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto elem_type =
      getInTensor(LSTMGradOp::getOutputGradInIndex()).elementType();

  if (lstm_grad_op.hasCellStateGradInput()) {
    return getInTensor(LSTMGradOp::getCellStateOutputGradInIndex())
        .reshape({batch_size, hidden_size});
  } else {
    auto zero = getScalarVariable(elem_type, "lstm/zero_cell_state");
    graph().setTileMapping(zero, 0);
    graph().setInitialValue(zero, 0);
    zero = zero.expand({0, 0});
    zero = zero.broadcast(batch_size, 0);
    zero = zero.broadcast(hidden_size, 1);
    return zero;
  }
}

poplar::Tensor LSTMGradOpx::getHiddenStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();
  auto &lstm_op      = lstm_grad_op.getForwardOp();

  unsigned batch_size  = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto elem_type =
      getInTensor(LSTMGradOp::getOutputGradInIndex()).elementType();

  if (lstm_grad_op.hasHiddenStateGradInput()) {
    return getInTensor(LSTMGradOp::getHiddenStateOutputGradInIndex())
        .reshape({batch_size, hidden_size});
  } else {
    auto zero = getScalarVariable(elem_type, "lstm/zero_hidden_state");
    graph().setTileMapping(zero, 0);
    graph().setInitialValue(zero, 0);
    zero = zero.expand({0, 0});
    zero = zero.broadcast(batch_size, 0);
    zero = zero.broadcast(hidden_size, 1);
    return zero;
  }
}

popnn::lstm::LstmParams LSTMGradOpx::createLSTMParams() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();
  return LSTMOpx::createLSTMParams(lstm_grad_op.getForwardOp());
}

namespace {
OpxCreator<LSTMOpx> lstmOpxCreator({Onnx::Operators::LSTM_1,
                                    Onnx::Operators::LSTM_7});
OpxCreator<LSTMGradOpx> lstmGradOpxCreator(Onnx::GradOperators::LSTMGrad);
} // namespace

} // namespace popx
} // namespace popart
