// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/lstmx.hpp>
#include <popart/popx/op/lstmxutil.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

namespace {

poplar::Tensor *getPoplarTensor(snap::Tensor *t) {
  if (t) {
    return &t->getPoplarTensor();
  } else {
    return nullptr;
  }
}

} // unnamed namespace

LSTMOpx::LSTMOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<LSTMOp>(op, {Onnx::Operators::LSTM_1, Onnx::Operators::LSTM_7});
}

// Only create an intermediate tensor if it is consumed or used as a anchor
std::unique_ptr<snap::Tensor> LSTMOpx::createIntermediate() const {
  if (getOp<LSTMOp>().isTraining()) {
    return std::make_unique<snap::Tensor>(poplar::Tensor{}, graph());
  } else {
    return std::unique_ptr<snap::Tensor>(nullptr);
  }
}

void LSTMOpx::grow(snap::program::Sequence &prog) const {
  prepareWeights(prog);
  growBias(prog);

  auto init_state = getInitialState();
  prepareInitialState(init_state, prog);

  auto intermediate = createIntermediate();
  poplar::Tensor outputP, cell_stateP;
  auto input    = getInput(prog);
  auto &lstm_op = getOp<LSTMOp>();
  auto seq_lens = getSeqLens();
  std::tie(outputP, cell_stateP) =
      popnn::lstm::lstmFwd(graph().getPoplarGraph(),
                           createLSTMParams(lstm_op, seq_lens),
                           init_state,
                           input.getPoplarTensor(),
                           *weights,
                           getPoplarTensor(intermediate.get()),
                           prog.getPoplarSequence(),
                           debugContext("lstmFwd"),
                           dv_p->lowering().lstmOptions,
                           &dv_p->matmulCache);
  snap::Tensor output{outputP, graph()};
  snap::Tensor cell_state{cell_stateP, graph()};

  if (intermediate) {
    setOutTensor(LSTMOp::getIntermediatesPassThroughIndex(), *intermediate);
  }

  reshapeAndInsert(LSTMOp::getOutputOutIndex(), output);

  auto output_h_state =
      output[createLSTMParams(lstm_op, seq_lens).rnn.timeSteps - 1];

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  reshapeAndInsert(LSTMOp::getHiddenStateOutIndex(),
                   cloneNcopy(prog, output_h_state));
  reshapeAndInsert(LSTMOp::getCellStateOutIndex(), cell_state);

  setOutTensor(LSTMOp::getInitStateOutputPassThroughIndex(),
               snap::Tensor{init_state.output, graph()});
  setOutTensor(LSTMOp::getInitStateCellStatePassThroughIndex(),
               snap::Tensor{init_state.cellState, graph()});
  setOutTensor(LSTMOp::getInputWeightsPassThroughIndex(),
               snap::Tensor{weights->inputWeights, graph()});
  setOutTensor(LSTMOp::getOutputWeightsPassThroughIndex(),
               snap::Tensor{weights->outputWeights, graph()});
  setOutTensor(LSTMOp::getBiasesPassThroughIndex(),
               snap::Tensor{weights->biases, graph()});

  // TODO T18126 register this alias or insert a cloneNcopy
  setOutTensor(LSTMOp::getInputPassThroughIndex(), input);

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  setOutTensor(LSTMOp::getOutputPassThroughIndex(), cloneNcopy(prog, output));
}

void LSTMOpx::reshapeAndInsert(OutIndex index,
                               const snap::Tensor &tensor) const {
  if (getOp<LSTMOp>().hasOutput(index)) {
    setOutTensor(index, tensor.reshape(outInfo(index).shape_szt()));
  }
}

snap::Tensor LSTMOpx::getSeqLens() const {
  if (hasInput(LSTMOp::getSequenceLensInIndex())) {
    return getInTensor(LSTMOp::getSequenceLensInIndex())
        .reinterpret(poplar::UNSIGNED_INT);
  } else {
    return snap::Tensor(poplar::Tensor{}, graph());
  }
}

void LSTMOpx::growBias(snap::program::Sequence &prog) const {
  // bias in onnx is shape [num_directions, 8 * hidden_size]
  // bias in poplibs is [4, hidden_size]
  auto &lstm_op        = getOp<LSTMOp>();
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(
      snap::Tensor{getLSTMWeights().biases, graph()}, false);

  if (lstm_op.hasBiasInput()) {
    auto bias_input = getInTensor(LSTMOp::getBiasInIndex()).getPoplarTensor();

    poplar::program::Copy copyProg(bias_input.slice(0, 4 * hidden_size, 1),
                                   biases.getPoplarTensor(),
                                   false,
                                   debugContext());
    prog.add(copyProg);

    popops::mapInPlace(graph().getPoplarGraph(),
                       popops::expr::BinaryOpType::ADD,
                       biases.getPoplarTensor(),
                       bias_input.slice(4 * hidden_size, 8 * hidden_size, 1),
                       prog.getPoplarSequence(),
                       debugContext("add"));
  } else {
    popops::zero(graph().getPoplarGraph(),
                 biases.getPoplarTensor(),
                 prog.getPoplarSequence(),
                 debugContext("zero"));
  }
}

InputCreatorType LSTMOpx::getInputCreatorType(InIndex index) const {
  if (index == LSTMOp::getInputInIndex() ||
      index == LSTMOp::getWeightsInIndex() ||
      index == LSTMOp::getRecurrenceInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

snap::Tensor
LSTMOpx::createInputTensor(InIndex index,
                           const poplar::DebugNameAndId &dnai) const {
  createdInputs.insert(index);

  if (index == LSTMOp::getInputInIndex()) {
    return createLSTMInput();
  } else if (index == LSTMOp::getWeightsInIndex()) {
    auto inputWeights = snap::Tensor{getLSTMWeights().inputWeights, graph()};
    return reshapePoplibWeightsForOnnx(inputWeights, true);
  } else if (index == LSTMOp::getRecurrenceInIndex()) {
    auto outputWeights = snap::Tensor{getLSTMWeights().outputWeights, graph()};
    return reshapePoplibWeightsForOnnx(outputWeights, true);
  } else {
    throw error("LSTMOpx::createInput is not supported for index {}", index);
  }
}

bool LSTMOpx::inputCreated(InIndex index) const {
  return createdInputs.count(index) > 0;
}

snap::Tensor LSTMOpx::reshapePoplibWeightsForOnnx(snap::Tensor poplib_weights,
                                                  bool transpose) {
  // ONNX expects input weights in shape [num_directions, 4*hidden_size, K]
  // where
  //   num_directions is always 1 for popart
  //   and K is either input_size or hidden_size, for the inputWeights or
  //   outputWeights respectively
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

  auto wf = slices[0].getPoplarTensor();
  auto wi = slices[1].getPoplarTensor();
  auto wc = slices[2].getPoplarTensor();
  auto wo = slices[3].getPoplarTensor();

  return snap::Tensor{poplar::concat({wi, wo, wf, wc}, 1), poplib_weights};
}

snap::Tensor LSTMOpx::createLSTMInput() const {
  auto &lstm_op    = getOp<LSTMOp>();
  auto seq_lens    = getSeqLens();
  auto lstm_params = createLSTMParams(lstm_op, seq_lens);
  auto options     = dv_p->lowering().lstmOptions;
  auto cache       = &dv_p->matmulCache;

  return snap::Tensor{popnn::lstm::createInput(graph().getPoplarGraph(),
                                               lstm_params,
                                               getDebugNameAndId("input"),
                                               options,
                                               cache),
                      graph()};
}

popnn::lstm::LstmState LSTMOpx::getInitialState() const {
  if (!initial_state) {
    auto options     = dv_p->lowering().lstmOptions;
    auto cache       = &dv_p->matmulCache;
    auto &lstm_op    = getOp<LSTMOp>();
    auto seq_lens    = getSeqLens();
    auto lstm_params = createLSTMParams(lstm_op, seq_lens);

    initial_state = createInitialState(graph().getPoplarGraph(),
                                       lstm_params,
                                       getDebugNameAndId("initialState"),
                                       options,
                                       cache);
  }

  return *initial_state;
}

popnn::lstm::LstmWeights LSTMOpx::getLSTMWeights() const {
  if (!weights) {
    auto &lstm_op    = getOp<LSTMOp>();
    auto seq_lens    = getSeqLens();
    auto lstm_params = createLSTMParams(lstm_op, seq_lens);
    auto options     = dv_p->lowering().lstmOptions;
    auto cache       = &dv_p->matmulCache;

    weights = popnn::lstm::createWeights(graph().getPoplarGraph(),
                                         lstm_params,
                                         debugContext("weights"),
                                         options,
                                         cache);
  }

  return *weights;
}

popnn::lstm::LstmParams
LSTMOpx::createLSTMParams(const LSTMOp &lstm_op,
                          const snap::Tensor &seq_lens_t) {
  auto in_info         = lstm_op.inInfo(LSTMOp::getInputInIndex());
  auto max_seq_length  = static_cast<unsigned>(lstm_op.getMaxSeqLength());
  auto batch_size      = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned input_size  = static_cast<unsigned>(lstm_op.getInputSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  if (seq_lens_t.valid()) {
    return popnn::lstm::LstmParams(popType(in_info),
                                   batch_size,
                                   max_seq_length,
                                   seq_lens_t.getPoplarTensor(),
                                   {input_size, hidden_size},
                                   convert(lstm_op.getActivation()),
                                   convert(lstm_op.getRecurrentActivation()));
  }
  return popnn::lstm::LstmParams(popType(in_info),
                                 batch_size,
                                 max_seq_length,
                                 {input_size, hidden_size},
                                 convert(lstm_op.getActivation()),
                                 convert(lstm_op.getRecurrentActivation()));
}

std::set<TensorId> LSTMOpx::mustExistBeforeCreate(InIndex) const {
  if (getOp<LSTMOp>().hasSeqLenInput()) {
    return {inId(LSTMOp::getSequenceLensInIndex())};
  }
  return {};
}

void LSTMOpx::prepareWeights(snap::program::Sequence &prog) const {
  // check to see if the weights were created
  prog.add(poplar::program::Copy(
      getInTensor(LSTMOp::getWeightsInIndex()).getPoplarTensor(),
      reshapePoplibWeightsForOnnx(
          snap::Tensor{getLSTMWeights().inputWeights, graph()}, true)
          .getPoplarTensor(),
      false,
      debugContext()));
  prog.add(poplar::program::Copy(
      getInTensor(LSTMOp::getRecurrenceInIndex()).getPoplarTensor(),
      reshapePoplibWeightsForOnnx(
          snap::Tensor{getLSTMWeights().outputWeights, graph()}, true)
          .getPoplarTensor(),
      false,
      debugContext()));
}

snap::Tensor LSTMOpx::getInput(snap::program::Sequence &prog) const {
  if (!inputCreated(LSTMOp::getInputInIndex())) {
    auto input     = createInputTensor(LSTMOp::getInputInIndex(),
                                   getDebugNameAndId("input"));
    auto raw_input = getInTensor(LSTMOp::getInputInIndex());
    prog.add(poplar::program::Copy(raw_input.getPoplarTensor(),
                                   input.getPoplarTensor(),
                                   false,
                                   debugContext()));
    return input;
  } else {
    return getInTensor(LSTMOp::getInputInIndex());
  }
}

void LSTMOpx::prepareInitialState(popnn::lstm::LstmState &init_state,
                                  snap::program::Sequence &prog) const {
  auto &lstm_op           = getOp<LSTMOp>();
  auto hasInitC           = lstm_op.hasInitialCInput();
  auto hasInitH           = lstm_op.hasInitialHInput();
  unsigned batch_size     = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned hidden_size    = static_cast<unsigned>(lstm_op.getHiddenSize());
  unsigned num_directions = static_cast<unsigned>(lstm_op.getNumDirections());

  // If initC and initH are not present, one or both will need zeroing.
  if (!hasInitC && !hasInitH) {
    zeroInitialState(graph().getPoplarGraph(),
                     init_state,
                     prog.getPoplarSequence(),
                     debugContext());
  } else if (!hasInitC) {
    popops::zero(graph().getPoplarGraph(),
                 init_state.cellState,
                 prog.getPoplarSequence(),
                 debugContext());
  } else if (!hasInitH) {
    popops::zero(graph().getPoplarGraph(),
                 init_state.output,
                 prog.getPoplarSequence(),
                 debugContext());
  }

  // Copy initC input to initialState.cellState is initC is provided.
  if (hasInitC) {
    auto init_c = getInitialState().cellState;
    init_c      = init_c.reshape({num_directions, batch_size, hidden_size});

    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getInitialCInIndex()).getPoplarTensor(),
        init_c,
        false,
        debugContext()));
  }
  // Copy initH input to initialState.output is initH is provided.
  if (hasInitH) {
    auto init_h = getInitialState().output;
    init_h      = init_h.reshape({num_directions, batch_size, hidden_size});

    prog.add(poplar::program::Copy(
        getInTensor(LSTMOp::getInitialHInIndex()).getPoplarTensor(),
        init_h,
        false,
        debugContext()));
  }
}

LSTMGradOpx::LSTMGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<LSTMGradOp>(op, Onnx::GradOperators::LSTMGrad);
}

void LSTMGradOpx::grow(snap::program::Sequence &prog) const {
  popnn::lstm::LstmState init_state;
  init_state.output =
      getInTensor(LSTMGradOp::getInitStateOutputInIndex()).getPoplarTensor();
  init_state.cellState =
      getInTensor(LSTMGradOp::getInitStateCellStateInIndex()).getPoplarTensor();

  popnn::lstm::LstmWeights weights;
  weights.inputWeights =
      getInTensor(LSTMGradOp::getInputWeightsInIndex()).getPoplarTensor();
  weights.outputWeights =
      getInTensor(LSTMGradOp::getOutputWeightsInIndex()).getPoplarTensor();
  weights.biases =
      getInTensor(LSTMGradOp::getBiasesInIndex()).getPoplarTensor();

  auto intermediates =
      getInTensor(LSTMGradOp::getIntermediatesInIndex()).getPoplarTensor();
  auto forward_input =
      getInTensor(LSTMGradOp::getInputInIndex()).getPoplarTensor();
  auto forward_output =
      getInTensor(LSTMGradOp::getOutputInIndex()).getPoplarTensor();

  auto &lstm_grad_op  = getOp<LSTMGradOp>();
  auto &lstm_op       = lstm_grad_op.getForwardOp();
  auto batch_size     = static_cast<unsigned>(lstm_op.getBatchSize());
  auto hidden_size    = static_cast<unsigned>(lstm_op.getHiddenSize());
  auto max_seq_length = static_cast<unsigned>(lstm_op.getMaxSeqLength());
  auto num_directions = static_cast<unsigned>(lstm_op.getNumDirections());

  auto seq_lens    = getSeqLens();
  auto lstm_params = LSTMOpx::createLSTMParams(lstm_op, seq_lens);

  auto output_grad = getInTensor(LSTMGradOp::getOutputGradInIndex())
                         .getPoplarTensor()
                         .reshape({max_seq_length, batch_size, hidden_size});

  auto output_c_grad = getCellStateGrad();
  auto output_h_grad = getHiddenStateGrad();

  // TODO find out what this is for
  // it's done in tensorflow and enigma
  auto output_grad_copy =
      cloneNcopy(prog, snap::Tensor{output_grad, graph()}).getPoplarTensor();
  popops::addInPlace(graph().getPoplarGraph(),
                     output_grad_copy[output_grad_copy.dim(0) - 1],
                     output_h_grad.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     debugContext());

  poplar::Tensor input_grad;
  popnn::lstm::LstmWeights weights_grad;

  if (lstm_params.rnn.variableTimeSteps() &&
      lstm_grad_op.hasCellStateGradInput()) {
    logging::opx::warn(
        "Looks like you are attempting to use the cell state output (LSTMOp "
        "output Y_c) and the sequence lengths input (LSTMOp input "
        "sequence_lens) of the LSTMOp at the same time, for the op {}. This is "
        "no longer supported and the cell state gradient shall just be treated "
        "as a tensor of zeros",
        lstm_op.debugName());
  }

  popnn::lstm::LstmState lastStepStateGrad;
  popnn::lstm::LstmState *lastStepStateGradPtr = nullptr;
  if (!lstm_params.rnn.variableTimeSteps()) {
    lastStepStateGrad.cellState = output_c_grad.getPoplarTensor();
    lastStepStateGradPtr        = &lastStepStateGrad;
  }

  auto init_state_grad = lstmBwdWithWU(graph().getPoplarGraph(),
                                       lstm_params,
                                       prog.getPoplarSequence(),
                                       init_state,
                                       intermediates,
                                       weights,
                                       forward_input,
                                       forward_output,
                                       output_grad_copy,
                                       lastStepStateGradPtr,
                                       &input_grad,
                                       weights_grad,
                                       debugContext("lstmBwdWithWU"),
                                       dv_p->lowering().lstmOptions,
                                       &dv_p->matmulCache);

  setOutTensor(LSTMGradOp::getInputOutIndex(),
               snap::Tensor{input_grad, graph()});
  setOutTensor(LSTMGradOp::getWeightsOutIndex(),
               LSTMOpx::reshapePoplibWeightsForOnnx(
                   snap::Tensor{weights_grad.inputWeights, graph()}, true));
  setOutTensor(LSTMGradOp::getRecurrenceOutIndex(),
               LSTMOpx::reshapePoplibWeightsForOnnx(
                   snap::Tensor{weights_grad.outputWeights, graph()}, true));

  if (lstm_op.hasBiasInput()) {
    auto b_grad = LSTMOpx::reshapePoplibWeightsForOnnx(
        snap::Tensor{weights_grad.biases, graph()}, false);
    setOutTensor(LSTMGradOp::getBiasOutIndex(),
                 snap::Tensor{poplar::concat({b_grad.getPoplarTensor(),
                                              b_grad.getPoplarTensor()},
                                             1),
                              graph()});
  }
  if (lstm_op.hasInitialHInput()) {
    auto init_h = init_state_grad.output;
    setOutTensor(
        LSTMGradOp::getInitialHOutIndex(),
        snap::Tensor{init_h.reshape({num_directions, batch_size, hidden_size}),
                     graph()});
  }
  if (lstm_op.hasInitialCInput()) {
    auto init_c = init_state_grad.cellState;
    setOutTensor(
        LSTMGradOp::getInitialCOutIndex(),
        snap::Tensor{init_c.reshape({num_directions, batch_size, hidden_size}),
                     graph()});
  }
}

snap::Tensor LSTMGradOpx::getCellStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();
  auto &lstm_op      = lstm_grad_op.getForwardOp();

  unsigned batch_size  = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto elem_type = getInTensor(LSTMGradOp::getOutputGradInIndex())
                       .getPoplarTensor()
                       .elementType();

  if (lstm_grad_op.hasCellStateGradInput()) {
    return snap::Tensor{getInTensor(LSTMGradOp::getCellStateOutputGradInIndex())
                            .getPoplarTensor()
                            .reshape({batch_size, hidden_size}),
                        graph()};
  } else {
    auto zero =
        getScalarVariable(elem_type, "lstm/zero_cell_state").getPoplarTensor();
    graph().getPoplarGraph().setTileMapping(zero, 0);
    graph().getPoplarGraph().setInitialValue(zero, 0);
    zero = zero.expand({0, 0});
    zero = zero.broadcast(batch_size, 0);
    zero = zero.broadcast(hidden_size, 1);
    return snap::Tensor{zero, graph()};
  }
}

snap::Tensor LSTMGradOpx::getSeqLens() const {
  if (hasInput(LSTMGradOp::getSequenceLensInIndex())) {
    return getInTensor(LSTMGradOp::getSequenceLensInIndex())
        .reinterpret(poplar::UNSIGNED_INT);
  } else {
    return snap::Tensor();
  }
}

snap::Tensor LSTMGradOpx::getHiddenStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();
  auto &lstm_op      = lstm_grad_op.getForwardOp();

  unsigned batch_size  = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto elem_type = getInTensor(LSTMGradOp::getOutputGradInIndex())
                       .getPoplarTensor()
                       .elementType();

  if (lstm_grad_op.hasHiddenStateGradInput()) {
    return snap::Tensor{
        getInTensor(LSTMGradOp::getHiddenStateOutputGradInIndex())
            .getPoplarTensor()
            .reshape({batch_size, hidden_size}),
        graph()};
  } else {
    auto zero = getScalarVariable(elem_type, "lstm/zero_hidden_state")
                    .getPoplarTensor();
    graph().getPoplarGraph().setTileMapping(zero, 0);
    graph().getPoplarGraph().setInitialValue(zero, 0);
    zero = zero.expand({0, 0});
    zero = zero.broadcast(batch_size, 0);
    zero = zero.broadcast(hidden_size, 1);
    return snap::Tensor{zero, graph()};
  }
}

namespace {
OpxCreator<LSTMOpx> lstmOpxCreator({Onnx::Operators::LSTM_1,
                                    Onnx::Operators::LSTM_7});
OpxCreator<LSTMGradOpx> lstmGradOpxCreator(Onnx::GradOperators::LSTMGrad);
} // namespace

} // namespace popx
} // namespace popart
