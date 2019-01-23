#include <spdlog/fmt/fmt.h>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/lstm.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/lstmx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace poponnx {
namespace popx {

LSTMOpx::LSTMOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LSTMOp>(op, {Onnx::Operators::LSTM_1, Onnx::Operators::LSTM_7});
}

// Only create an intermediate tensor if it is consumed or used as a anchor
std::unique_ptr<poplar::Tensor> LSTMOpx::createIntermediate() const {
  if (outTensor(LSTMOp::getOutputOutIndex())->consumers.getTotal() > 0) {
    return make_unique<poplar::Tensor>();
  }

  auto anchors = op_p->getIr().getDataFlow().anchors();
  if (std::find(anchors.begin(),
                anchors.end(),
                outId(LSTMOp::getOutputOutIndex())) != anchors.end()) {
    return make_unique<poplar::Tensor>();
  }

  return std::unique_ptr<poplar::Tensor>(nullptr);
}

void LSTMOpx::grow(poplar::program::Sequence &prog) const {
  growBias(prog);

  auto init_state = getInitialState();
  prepareInitialState(init_state, prog);

  auto intermediate = createIntermediate();
  poplar::Tensor output, cell_state;
  auto input                   = get(inId(LSTMOp::getInputInIndex()));
  std::tie(output, cell_state) = popnn::lstm::lstmFwd(graph(),
                                                      createLSTMParams(),
                                                      init_state,
                                                      input,
                                                      *weights,
                                                      intermediate.get(),
                                                      prog,
                                                      idStr(),
                                                      dv_p->lstmOptions,
                                                      &dv_p->matmulCache);

  if (intermediate) {
    insert(outId(LSTMOp::getIntermediatesPassThroughIndex()), *intermediate);
  }

  reshapeAndInsert(LSTMOp::getOutputOutIndex(), output);

  auto output_h_state = output[createLSTMParams().timeSteps - 1];
  reshapeAndInsert(LSTMOp::getHiddenStateOutIndex(), output_h_state);
  reshapeAndInsert(LSTMOp::getCellStateOutIndex(), cell_state);

  insert(outId(LSTMOp::getInitStateOutputPassThroughIndex()),
         init_state.output);
  insert(outId(LSTMOp::getInitStateCellStatePassThroughIndex()),
         init_state.cellState);
  insert(outId(LSTMOp::getInputWeightsPassThroughIndex()),
         weights->inputWeights);
  insert(outId(LSTMOp::getOutputWeightsPassThroughIndex()),
         weights->outputWeights);
  insert(outId(LSTMOp::getBiasesPassThroughIndex()), weights->biases);
  insert(outId(LSTMOp::getInputPassThroughIndex()), input);
  insert(outId(LSTMOp::getOutputPassThroughIndex()), output);
}

void LSTMOpx::reshapeAndInsert(OutIndex index,
                               const poplar::Tensor &tensor) const {
  insert(outId(index), tensor.reshape(outInfo(index).shape_szt()));
}

void LSTMOpx::growBias(poplar::program::Sequence &prog) const {
  // bias in onnx is shape [num_directions, 8 * hidden_size]
  // bias in poplibs is [4, hidden_size]
  auto &lstm_op        = getOp<LSTMOp>();
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(getLSTMWeights().biases, false);

  if (lstm_op.hasBiasInput()) {
    auto bias_input = get(inId(LSTMOp::getBiasInIndex()));

    poplar::program::Copy copyProg(bias_input.slice(0, 4 * hidden_size, 1),
                                   biases);
    prog.add(copyProg);

    popops::mapInPlace(graph(),
                       popops::expr::BinaryOpType::ADD,
                       biases,
                       bias_input.slice(4 * hidden_size, 8 * hidden_size, 1),
                       prog,
                       idStr());
  } else {
    popops::zero(graph(), biases, prog, idStr());
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

poplar::Tensor LSTMOpx::createInput(InIndex index) const {
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
    return getInitialState().output;
  } else {
    auto msg = fmt::format("LSTMOpx::createInput is not supported for index {}",
                           index);
    throw error(msg);
  }
}

poplar::Tensor
LSTMOpx::reshapePoplibWeightsForOnnx(poplar::Tensor poplib_weights,
                                     bool transpose) {
  // ONNX expects input weights in shape [num_directions, 4*hidden_size, K]
  // where
  //   num_directions is always 1 for poponnx
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
      graph(), lstm_params, idStr(), options, cache);
}

popnn::lstm::LstmState LSTMOpx::getInitialState() const {
  if (!initial_state) {
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;
    auto lstm_params = createLSTMParams();

    initial_state =
        createInitialState(graph(), lstm_params, idStr(), options, cache);
  }

  return *initial_state;
}

popnn::lstm::LstmWeights LSTMOpx::getLSTMWeights() const {
  if (!weights) {
    auto lstm_params = createLSTMParams();
    auto options     = dv_p->lstmOptions;
    auto cache       = &dv_p->matmulCache;

    weights = createWeights(graph(), lstm_params, idStr(), options, cache);
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

void LSTMOpx::prepareInitialState(popnn::lstm::LstmState &init_state,
                                  poplar::program::Sequence &prog) const {
  auto hasInitC = op_p->input->hasIndex(LSTMOp::getInitialCInIndex());
  auto hasInitH = op_p->input->hasIndex(LSTMOp::getInitialHInIndex());

  if (!hasInitC && !hasInitH) {
    zeroInitialState(graph(), init_state, prog, idStr());
  } else if (!hasInitC) {
    popops::zero(graph(), init_state.cellState, prog, idStr());
  } else if (!hasInitH) {
    popops::zero(graph(), init_state.output, prog, idStr());
  }
}

LSTMGradOpx::LSTMGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LSTMGradOp>(op, Onnx::GradOperators::LSTMGrad);
}

void LSTMGradOpx::grow(poplar::program::Sequence &prog) const {
  popnn::lstm::LstmState init_state;
  init_state.output    = get(inId(LSTMGradOp::getInitStateOutputInIndex()));
  init_state.cellState = get(inId(LSTMGradOp::getInitStateCellStateInIndex()));

  popnn::lstm::LstmWeights weights;
  weights.inputWeights  = get(inId(LSTMGradOp::getInputWeightsInIndex()));
  weights.outputWeights = get(inId(LSTMGradOp::getOutputWeightsInIndex()));
  weights.biases        = get(inId(LSTMGradOp::getBiasesInIndex()));

  auto intermediates  = get(inId(LSTMGradOp::getIntermediatesInIndex()));
  auto forward_input  = get(inId(LSTMGradOp::getInputInIndex()));
  auto forward_output = get(inId(LSTMGradOp::getOutputInIndex()));

  auto &lstm_grad_op = getOp<LSTMGradOp>();
  auto &lstm_op      = lstm_grad_op.getForwardOp();
  auto batch_size    = static_cast<unsigned>(lstm_op.getBatchSize());
  auto hidden_size   = static_cast<unsigned>(lstm_op.getHiddenSize());
  auto seq_length    = static_cast<unsigned>(lstm_op.getSeqLength());
  auto lstm_params   = createLSTMParams();

  auto output_grad = get(inId(LSTMGradOp::getOutputGradInIndex()))
                         .reshape({seq_length, batch_size, hidden_size});
  auto output_c_grad = get(inId(LSTMGradOp::getCellStateOutputGradInIndex()))
                           .reshape({batch_size, hidden_size});
  auto output_h_grad = get(inId(LSTMGradOp::getHiddenStateOutputGradInIndex()))
                           .reshape({batch_size, hidden_size});

  // TODO find out what this is for
  // it's done in tensorflow and enigma
  auto output_grad_copy = cloneNcopy(prog, output_grad);
  popops::addInPlace(graph(),
                     output_grad_copy[output_grad_copy.dim(0) - 1],
                     output_h_grad,
                     prog,
                     idStr());

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
                                       idStr(),
                                       dv_p->lstmOptions,
                                       &dv_p->matmulCache);

  insert(outId(LSTMGradOp::getInputOutIndex()), input_grad);
  insert(outId(LSTMGradOp::getWeightsOutIndex()),
         LSTMOpx::reshapePoplibWeightsForOnnx(weights_grad.inputWeights, true));
  insert(
      outId(LSTMGradOp::getRecurrenceOutIndex()),
      LSTMOpx::reshapePoplibWeightsForOnnx(weights_grad.outputWeights, true));

  if (lstm_op.hasBiasInput()) {
    auto b_grad = poplar::concat({weights_grad.biases, weights_grad.biases}, 1);
    insert(outId(LSTMGradOp::getBiasOutIndex()),
           LSTMOpx::reshapePoplibWeightsForOnnx(b_grad, false));
  }
  if (lstm_op.hasInitialHInput()) {
    insert(outId(LSTMGradOp::getInitialHOutIndex()), init_state_grad.output);
  }
  if (lstm_op.hasInitialCInput()) {
    insert(outId(LSTMGradOp::getInitialCOutIndex()), init_state_grad.cellState);
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
} // namespace poponnx
