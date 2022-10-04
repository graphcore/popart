// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <tuple>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popnn/Lstm.hpp>
#include <popnn/Rnn.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/lstmx.hpp>
#include <popart/popx/op/lstmxutil.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class Op;

namespace popx {

LSTMOpx::LSTMOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
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

static poplar::OptionFlags
addAvailableMemoryProportionOption(const LSTMOp &op,
                                   const poplar::OptionFlags &flags) {
  auto rflags     = flags;
  const auto &amp = op.getAvailableMemoryProportion();
  if (amp.has_value()) {
    rflags.set("availableMemoryProportion", std::to_string(*amp));
  }
  return rflags;
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
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);

  std::tie(outputP, cell_stateP) =
      popnn::lstm::lstmFwd(graph().getPoplarGraph(),
                           createLSTMParams(),
                           init_state,
                           input.getPoplarTensor(),
                           *weights,
                           intermediate.get(),
                           prog.getPoplarSequence(),
                           debugContext("lstmFwd"),
                           options,
                           &dv_p->matmulCache);
  snap::Tensor output{outputP, graph()};
  snap::Tensor cell_state{cell_stateP, graph()};

  if (intermediate) {
    setOutTensor(LSTMOp::getIntermediatesPassThroughIndex(),
                 snap::Tensor{*intermediate, graph()});
  }

  reshapeAndInsert(LSTMOp::getFullHiddenStateOutIndex(), output);

  auto output_h_state = output[createLSTMParams().rnn.timeSteps - 1];

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  reshapeAndInsert(LSTMOp::getLastHiddenStateOutIndex(),
                   cloneNcopy(prog, output_h_state));
  reshapeAndInsert(LSTMOp::getLastCellStateOutIndex(), cell_state);

  setOutTensor(LSTMOp::getInitialHPassThroughIndex(),
               snap::Tensor{init_state.output, graph()});
  setOutTensor(LSTMOp::getInitialCPassThroughIndex(),
               snap::Tensor{init_state.cellState, graph()});
  setOutTensor(LSTMOp::getInputWeightsPassThroughIndex(),
               snap::Tensor{weights->inputWeights, graph()});
  setOutTensor(LSTMOp::getRecurrenceWeightsPassThroughIndex(),
               snap::Tensor{weights->outputWeights, graph()});
  setOutTensor(LSTMOp::getBiasesPassThroughIndex(),
               snap::Tensor{weights->biases, graph()});
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
    return snap::Tensor{};
  }
}

void LSTMOpx::growBias(snap::program::Sequence &prog) const {
  // bias in onnx is shape [num_directions, 8 * hidden_size]
  // bias in poplibs is [4, hidden_size]
  auto &lstm_op        = getOp<LSTMOp>();
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(
      snap::Tensor{getLSTMWeights().biases, graph()}, false);

  if (lstm_op.hasBiasesInput()) {
    auto bias_input = getInTensor(LSTMOp::getBiasesInIndex());

    snap::program::Copy copyProg(
        bias_input.slice(0, 4 * hidden_size, 1), biases, false, debugContext());
    prog.getPoplarSequence().add(copyProg);

    snap::popops::mapInPlace(
        graph(),
        popops::expr::BinaryOpType::ADD,
        biases,
        bias_input.slice(4 * hidden_size, 8 * hidden_size, 1),
        prog,
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
      index == LSTMOp::getInputWeightsInIndex() ||
      index == LSTMOp::getRecurrenceWeightsInIndex()) {
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
  } else if (index == LSTMOp::getInputWeightsInIndex()) {
    auto inputWeights = snap::Tensor{getLSTMWeights().inputWeights, graph()};
    return reshapePoplibWeightsForOnnx(inputWeights, true);
  } else if (index == LSTMOp::getRecurrenceWeightsInIndex()) {
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
  auto lstm_params = createLSTMParams();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);
  auto cache = &dv_p->matmulCache;

  return snap::Tensor{popnn::lstm::createInput(graph().getPoplarGraph(),
                                               lstm_params,
                                               getDebugNameAndId("input"),
                                               options,
                                               cache),
                      graph()};
}

popnn::lstm::LstmState LSTMOpx::getInitialState() const {
  if (!initial_state) {
    auto cache       = &dv_p->matmulCache;
    auto &lstm_op    = getOp<LSTMOp>();
    auto lstm_params = createLSTMParams();
    auto options     = addAvailableMemoryProportionOption(
        lstm_op, dv_p->lowering().lstmOptions);

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
    auto lstm_params = createLSTMParams();
    auto options     = addAvailableMemoryProportionOption(
        lstm_op, dv_p->lowering().lstmOptions);
    auto cache = &dv_p->matmulCache;

    weights = popnn::lstm::createWeights(graph().getPoplarGraph(),
                                         lstm_params,
                                         debugContext("weights"),
                                         options,
                                         cache);
  }

  return *weights;
}

popnn::lstm::LstmParams LSTMOpx::createLSTMParams() const {
  auto &lstm_op        = getOp<LSTMOp>();
  auto in_info         = lstm_op.inInfo(LSTMOp::getInputInIndex());
  auto max_seq_length  = static_cast<unsigned>(lstm_op.getMaxSeqLength());
  auto batch_size      = static_cast<unsigned>(lstm_op.getBatchSize());
  unsigned input_size  = static_cast<unsigned>(lstm_op.getInputSize());
  unsigned hidden_size = static_cast<unsigned>(lstm_op.getHiddenSize());
  auto seq_lens_t      = getSeqLens();

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
  prog.getPoplarSequence().add(snap::program::Copy(
      getInTensor(LSTMOp::getInputWeightsInIndex()),
      reshapePoplibWeightsForOnnx(
          snap::Tensor{getLSTMWeights().inputWeights, graph()}, true),
      false,
      debugContext()));
  prog.getPoplarSequence().add(snap::program::Copy(
      getInTensor(LSTMOp::getRecurrenceWeightsInIndex()),
      reshapePoplibWeightsForOnnx(
          snap::Tensor{getLSTMWeights().outputWeights, graph()}, true),
      false,
      debugContext()));
}

snap::Tensor LSTMOpx::getInput(snap::program::Sequence &prog) const {
  if (!inputCreated(LSTMOp::getInputInIndex())) {
    auto input     = createInputTensor(LSTMOp::getInputInIndex(),
                                   getDebugNameAndId("input"));
    auto raw_input = getInTensor(LSTMOp::getInputInIndex());
    prog.getPoplarSequence().add(
        snap::program::Copy(raw_input, input, false, debugContext()));
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
    auto init_c = snap::Tensor{getInitialState().cellState, graph()};
    init_c      = init_c.reshape({num_directions, batch_size, hidden_size});

    prog.getPoplarSequence().add(
        snap::program::Copy(getInTensor(LSTMOp::getInitialCInIndex()),
                            init_c,
                            false,
                            debugContext()));
  }
  // Copy initH input to initialState.output is initH is provided.
  if (hasInitH) {
    auto init_h = snap::Tensor{getInitialState().output, graph()};
    init_h      = init_h.reshape({num_directions, batch_size, hidden_size});

    prog.getPoplarSequence().add(
        snap::program::Copy(getInTensor(LSTMOp::getInitialHInIndex()),
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
      getInTensor(LSTMGradOp::getInitialHInIndex()).getPoplarTensor();
  init_state.cellState =
      getInTensor(LSTMGradOp::getInitialCInIndex()).getPoplarTensor();

  popnn::lstm::LstmWeights weights;
  weights.inputWeights =
      getInTensor(LSTMGradOp::getInputWeightsInIndex()).getPoplarTensor();
  weights.outputWeights =
      getInTensor(LSTMGradOp::getRecurrenceWeightsInIndex()).getPoplarTensor();
  weights.biases =
      getInTensor(LSTMGradOp::getBiasesInIndex()).getPoplarTensor();

  auto &lstm_grad_op  = getOp<LSTMGradOp>();
  auto batch_size     = static_cast<unsigned>(lstm_grad_op.batch_size);
  auto hidden_size    = static_cast<unsigned>(lstm_grad_op.hidden_size);
  auto max_seq_length = static_cast<unsigned>(lstm_grad_op.max_seq_length);
  auto num_directions = static_cast<unsigned>(lstm_grad_op.num_directions);

  auto intermediates =
      getInTensor(LSTMGradOp::getIntermediatesInIndex()).getPoplarTensor();
  auto forward_input =
      getInTensor(LSTMGradOp::getInputInIndex()).getPoplarTensor();
  auto forward_output = getInTensor(LSTMGradOp::getFullHiddenStateInIndex())
                            .getPoplarTensor()
                            .reshape({max_seq_length, batch_size, hidden_size});

  auto lstm_params = createLSTMParams();

  auto output_grad = getInTensor(LSTMGradOp::getFullHiddenStateGradInIndex())
                         .getPoplarTensor()
                         .reshape({max_seq_length, batch_size, hidden_size});

  auto output_c_grad = getCellStateGrad();
  auto output_h_grad = getHiddenStateGrad();

  auto output_grad_copy = cloneNcopy(prog, snap::Tensor{output_grad, graph()});
  snap::popops::addInPlace(graph(),
                           output_grad_copy[output_grad_copy.dim(0) - 1],
                           output_h_grad,
                           prog,
                           debugContext());

  poplar::Tensor input_grad;
  popnn::lstm::LstmWeights weights_grad;

  if (lstm_params.rnn.variableTimeSteps() &&
      lstm_grad_op.hasLastCellStateGradInput()) {
    logging::opx::warn(
        "Looks like you are attempting to use the cell state output (LSTMOp "
        "output Y_c) and the sequence lengths input (LSTMOp input "
        "sequence_lens) of the LSTMOp at the same time, for the op {}. This is "
        "no longer supported and the cell state gradient shall just be treated "
        "as a tensor of zeros",
        lstm_grad_op.fwd_debug_name);
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
                                       output_grad_copy.getPoplarTensor(),
                                       lastStepStateGradPtr,
                                       &input_grad,
                                       weights_grad,
                                       debugContext("lstmBwdWithWU"),
                                       dv_p->lowering().lstmOptions,
                                       &dv_p->matmulCache);

  setOutTensor(LSTMGradOp::getInputOutIndex(),
               snap::Tensor{input_grad, graph()});
  setOutTensor(LSTMGradOp::getInputWeightsOutIndex(),
               LSTMOpx::reshapePoplibWeightsForOnnx(
                   snap::Tensor{weights_grad.inputWeights, graph()}, true));
  setOutTensor(LSTMGradOp::getRecurrenceWeightsOutIndex(),
               LSTMOpx::reshapePoplibWeightsForOnnx(
                   snap::Tensor{weights_grad.outputWeights, graph()}, true));

  if (lstm_grad_op.hasBiasesInput) {
    auto b_grad = LSTMOpx::reshapePoplibWeightsForOnnx(
        snap::Tensor{weights_grad.biases, graph()}, false);
    setOutTensor(LSTMGradOp::getBiasesOutIndex(),
                 snap::Tensor{poplar::concat({b_grad.getPoplarTensor(),
                                              b_grad.getPoplarTensor()},
                                             1),
                              graph()});
  }
  if (lstm_grad_op.hasInitialHInput) {
    auto init_h = init_state_grad.output;
    setOutTensor(
        LSTMGradOp::getInitialHOutIndex(),
        snap::Tensor{init_h.reshape({num_directions, batch_size, hidden_size}),
                     graph()});
  }
  if (lstm_grad_op.hasInitialCInput) {
    auto init_c = init_state_grad.cellState;
    setOutTensor(
        LSTMGradOp::getInitialCOutIndex(),
        snap::Tensor{init_c.reshape({num_directions, batch_size, hidden_size}),
                     graph()});
  }
}

snap::Tensor LSTMGradOpx::getCellStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();

  unsigned batch_size  = static_cast<unsigned>(lstm_grad_op.batch_size);
  unsigned hidden_size = static_cast<unsigned>(lstm_grad_op.hidden_size);

  auto elem_type = getInTensor(LSTMGradOp::getFullHiddenStateGradInIndex())
                       .getPoplarTensor()
                       .elementType();

  if (lstm_grad_op.hasLastCellStateGradInput()) {
    return snap::Tensor{getInTensor(LSTMGradOp::getLastCellStateGradInIndex())
                            .getPoplarTensor()
                            .reshape({batch_size, hidden_size}),
                        graph()};
  } else {
    auto zero =
        getScalarVariable(elem_type, "lstm/zero_cell_state").getPoplarTensor();
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
    return snap::Tensor{};
  }
}

popnn::lstm::LstmParams LSTMGradOpx::createLSTMParams() const {
  auto &lstm_grad_op   = getOp<LSTMGradOp>();
  auto in_info         = lstm_grad_op.inInfo(LSTMGradOp::getInputInIndex());
  auto max_seq_length  = lstm_grad_op.max_seq_length;
  auto batch_size      = lstm_grad_op.batch_size;
  unsigned input_size  = lstm_grad_op.input_size;
  unsigned hidden_size = lstm_grad_op.hidden_size;
  auto seq_lens_t      = getSeqLens();

  if (seq_lens_t.valid()) {
    return popnn::lstm::LstmParams(popType(in_info),
                                   batch_size,
                                   max_seq_length,
                                   seq_lens_t.getPoplarTensor(),
                                   {input_size, hidden_size},
                                   convert(lstm_grad_op.activation),
                                   convert(lstm_grad_op.recurrent_activation));
  }
  return popnn::lstm::LstmParams(popType(in_info),
                                 batch_size,
                                 max_seq_length,
                                 {input_size, hidden_size},
                                 convert(lstm_grad_op.activation),
                                 convert(lstm_grad_op.recurrent_activation));
}

snap::Tensor LSTMGradOpx::getHiddenStateGrad() const {
  auto &lstm_grad_op = getOp<LSTMGradOp>();

  unsigned batch_size  = static_cast<unsigned>(lstm_grad_op.batch_size);
  unsigned hidden_size = static_cast<unsigned>(lstm_grad_op.hidden_size);

  auto elem_type = getInTensor(LSTMGradOp::getFullHiddenStateGradInIndex())
                       .getPoplarTensor()
                       .elementType();

  if (lstm_grad_op.hasLastHiddenStateGradInput()) {
    return snap::Tensor{getInTensor(LSTMGradOp::getLastHiddenStateGradInIndex())
                            .getPoplarTensor()
                            .reshape({batch_size, hidden_size}),
                        graph()};
  } else {
    auto zero = getScalarVariable(elem_type, "lstm/zero_hidden_state")
                    .getPoplarTensor();
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
