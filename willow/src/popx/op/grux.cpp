// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/gru.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/grux.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <popnn/Gru.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

GRUOpx::GRUOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<GRUOp>(op, {Onnx::Operators::GRU_3, Onnx::Operators::GRU_7});
}

// Only create an intermediate tensor if it is consumed or used as a anchor
std::unique_ptr<poplar::Tensor> GRUOpx::createIntermediate() const {
  if (getOp<GRUOp>().isTraining()) {
    return std::make_unique<poplar::Tensor>();
  } else {
    return std::unique_ptr<poplar::Tensor>(nullptr);
  }
}

poplar::Tensor GRUOpx::getInitialState() const {
  if (!initial_state_h) {
    initial_state_h =
        popnn::gru::createInitialState(graph().getPoplarGraph(),
                                       createGRUParams(),
                                       debugContext("initialState"),
                                       dv_p->lowering().lstmOptions,
                                       &dv_p->matmulCache);
  }
  return *initial_state_h;
}

void GRUOpx::prepareInitialState(poplar::Tensor &init_state_h,
                                 poplar::program::Sequence &prog) const {
  auto &gru_op  = getOp<GRUOp>();
  auto hasInitH = gru_op.hasInitialHInput();

  if (!hasInitH) {
    popops::zero(graph().getPoplarGraph(), init_state_h, prog, debugContext());
  }

  // Check the inputs have been created
  if (hasInitH) {
    prog.add(
        poplar::program::Copy(getInTensor(GRUOp::getInitialHInIndex()),
                              createInputTensor(GRUOp::getInitialHInIndex(),
                                                getDebugNameAndId("initH"))
                                  .getPoplarTensor(),
                              false,
                              debugContext()));
  }
}

void GRUOpx::grow(poplar::program::Sequence &prog) const {
  prepareWeights(prog);
  growBias(prog);

  auto &gru_op = getOp<GRUOp>();
  auto input   = getInput(prog);

  auto init_state_h = getInitialState();
  prepareInitialState(init_state_h, prog);

  auto intermediate = createIntermediate();
  poplar::Tensor output;

  if (gru_op.getDirectionAttribute() == "forward") {
    output = popnn::gru::gruFwd(graph().getPoplarGraph(),
                                createGRUParams(),
                                init_state_h,
                                input,
                                *weights,
                                intermediate.get(),
                                prog,
                                debugContext("gruFwd"),
                                dv_p->lowering().lstmOptions,
                                &dv_p->matmulCache);
  } else if (gru_op.getDirectionAttribute() == "backward") {
    output = popnn::gru::gruFwd(graph().getPoplarGraph(),
                                createGRUParams(),
                                init_state_h,
                                input.reverse(0),
                                *weights,
                                intermediate.get(),
                                prog,
                                debugContext("gruFwd"),
                                dv_p->lowering().lstmOptions,
                                &dv_p->matmulCache);
  } else if (gru_op.getDirectionAttribute() == "bidirectional") {
    // TODO: Add support for bidirectional GRU op.
    throw error(
        "Bidirectional GRU has not yet been implemented. This is for Op {}",
        gru_op.debugName());
  }
  if (intermediate) {
    setOutTensor(GRUOp::getIntermediatesPassThroughIndex(), *intermediate);
  }

  reshapeAndInsert(GRUOp::getOutputOutIndex(), output);

  auto output_h_state = output[createGRUParams().rnn.timeSteps - 1];

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  reshapeAndInsert(GRUOp::getHiddenStateOutIndex(),
                   cloneNcopy(prog, output_h_state));

  setOutTensor(GRUOp::getInitStateOutputPassThroughIndex(), init_state_h);

  setOutTensor(GRUOp::getInputWeightsPassThroughIndex(), weights->inputWeights);
  setOutTensor(GRUOp::getOutputWeightsPassThroughIndex(),
               weights->outputWeights);

  auto biases = weights->biases;
  if (gru_op.getLinearBeforeResetAttribute()) {
    unsigned hidden_size = gru_op.getHiddenSize();
    biases               = weights->biases.reshape({6, hidden_size});
  }
  setOutTensor(GRUOp::getBiasesPassThroughIndex(), biases);

  setOutTensor(GRUOp::getInputPassThroughIndex(), input);

  setOutTensor(GRUOp::getOutputPassThroughIndex(), output);
}

void GRUOpx::reshapeAndInsert(OutIndex index,
                              const poplar::Tensor &tensor) const {
  if (getOp<GRUOp>().hasOutput(index)) {
    setOutTensor(index, tensor.reshape(outInfo(index).shape_szt()));
  }
}

void GRUOpx::growBias(poplar::program::Sequence &prog) const {
  auto &gru_op         = getOp<GRUOp>();
  unsigned hidden_size = static_cast<unsigned>(gru_op.getHiddenSize());
  auto biases          = getGRUWeights().biases;

  if (!gru_op.hasBiasInput()) {
    popops::zero(graph().getPoplarGraph(), biases, prog, debugContext("zero"));
    return;
  }

  // Onnx format is [1, 6 * hidden_size]
  // First subtensors are the input biases [bz, br, bh]
  // Following are the hidden biases [bhz, bhr, bhh]
  auto bias_input =
      getInTensor(GRUOp::getBiasInIndex()).reshape({6, hidden_size});

  std::vector<poplar::Interval> intervals{
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
  auto slices = bias_input.slices(intervals, 0);

  auto bz = slices[0];
  auto br = slices[1];
  auto bh = slices[2];

  auto bhz = slices[3];
  auto bhr = slices[4];
  auto bhh = slices[5];

  // There are two bias formats in poplib, depending on how the reset
  // gate is applied. In both circumstances, gate ordering is [R, Z, H]

  if (gru_op.getLinearBeforeResetAttribute()) {
    // If the reset hgate is applied after the linear transformation
    // the bias tensor is of shape [3, 2, hidden_size]. Inner planes
    // contains the input and hidden biases for the same gate.
    bias_input = poplar::concat({br, bhr, bz, bhz, bh, bhh});
    bias_input = bias_input.reshape({3, 2, hidden_size});

    poplar::program::Copy copyProg(bias_input, biases, false, debugContext());
    prog.add(copyProg);
    return;
  }

  // If the reset gate is applied beofore the linear transformation,
  // the bias tensor shaope is [3, hidden_size]. It is sufficient
  // to add the inpput and hidden biases together.
  auto input_bias  = poplar::concat({br, bz, bh});
  auto hidden_bias = poplar::concat({bhr, bhz, bhh});

  poplar::program::Copy copyProg(input_bias, biases, false, debugContext());
  prog.add(copyProg);

  popops::mapInPlace(graph().getPoplarGraph(),
                     popops::expr::BinaryOpType::ADD,
                     biases,
                     hidden_bias,
                     prog,
                     debugContext("add"));
}

InputCreatorType GRUOpx::getInputCreatorType(InIndex index) const {
  if (index == GRUOp::getInputInIndex() ||
      index == GRUOp::getWeightsInIndex() ||
      index == GRUOp::getRecurrenceInIndex() ||
      index == GRUOp::getInitialHInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

snap::Tensor GRUOpx::createInputTensor(InIndex index,
                                       const poplar::DebugNameAndId &) const {
  createdInputs.insert(index);

  if (index == GRUOp::getInputInIndex()) {
    return snap::Tensor{createGRUInput(), graph()};
  } else if (index == GRUOp::getWeightsInIndex()) {
    auto inputWeights = getGRUWeights().inputWeights;
    return snap::Tensor{reshapePoplibWeightsForOnnx(inputWeights), graph()};
  } else if (index == GRUOp::getRecurrenceInIndex()) {
    auto outputWeights = getGRUWeights().outputWeights;
    return snap::Tensor{reshapePoplibWeightsForOnnx(outputWeights), graph()};
  } else if (index == GRUOp::getInitialHInIndex()) {
    auto &gru_op = getOp<GRUOp>();

    unsigned batch_size     = static_cast<unsigned>(gru_op.getBatchSize());
    unsigned hidden_size    = static_cast<unsigned>(gru_op.getHiddenSize());
    unsigned num_directions = static_cast<unsigned>(gru_op.getNumDirections());

    auto init_h = getInitialState();
    return snap::Tensor{
        init_h.reshape({num_directions, batch_size, hidden_size}), graph()};
  } else {
    throw error("GRUOpx::createInput is not supported for index {}", index);
  }
}

bool GRUOpx::inputCreated(InIndex index) const {
  return createdInputs.count(index) > 0;
}

poplar::Tensor
GRUOpx::reshapePoplibWeightsForOnnx(poplar::Tensor poplib_weights) {
  // ONNX expects input weights in shape [num_directions, 3*hidden_size, K]
  // where
  //   num_directions is always 1 for popart
  //   and K is either input_size or hidden_size, for the inputWeights or
  //   outputWeights respectively
  // and order is W[zrh]
  //
  // poplibs expects weights in shape [3, K, hidden_size]
  // and order is W[rzh]
  std::vector<poplar::Interval> intervals{{0, 1}, {1, 2}, {2, 3}};
  auto slices = poplib_weights.slices(intervals, 0);

  for (int i = 0; i < slices.size(); i++) {
    slices[i] = slices[i].dimShuffle({0, 2, 1});
  }

  auto wz = slices[0];
  auto wr = slices[1];
  auto wh = slices[2];

  return poplar::concat({wr, wz, wh}, 1);
}

poplar::Tensor
GRUOpx::reshapePoplibBiasesForOnnx(poplar::Tensor poplib_biases) {
  bool hidden_biases = poplib_biases.rank() > 2;

  // Poplib uses two different bias formats depending on how the
  // reset gate is computed. See commentary in growBias.

  if (!hidden_biases) {
    std::vector<poplar::Interval> intervals{{0, 1}, {1, 2}, {2, 3}};
    auto slices = poplib_biases.slices(intervals, 0);

    auto bz = slices[0];
    auto br = slices[1];
    auto bh = slices[2];

    return poplar::concat({br, bz, bh}, 1);
  }
  size_t hidden_size = poplib_biases.dim(2);
  poplib_biases      = poplib_biases.reshape({6, hidden_size});

  std::vector<poplar::Interval> intervals{
      {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
  auto slices = poplib_biases.slices(intervals, 0);

  auto bz = slices[0];
  auto br = slices[2];
  auto bh = slices[4];

  auto bhz = slices[1];
  auto bhr = slices[3];
  auto bhh = slices[5];

  return poplar::concat({br, bz, bh, bhr, bhz, bhh}, 1);
}

poplar::Tensor GRUOpx::createGRUInput() const {
  auto gru_params = createGRUParams();
  auto options    = dv_p->lowering().lstmOptions;
  auto cache      = &dv_p->matmulCache;

  return popnn::gru::createInput(graph().getPoplarGraph(),
                                 gru_params,
                                 getDebugNameAndId("input"),
                                 options,
                                 cache);
}

popnn::gru::GruWeights GRUOpx::getGRUWeights() const {
  if (!weights) {
    auto gru_params = createGRUParams();
    auto options    = dv_p->lowering().lstmOptions;
    auto cache      = &dv_p->matmulCache;

    weights = createWeights(graph().getPoplarGraph(),
                            gru_params,
                            debugContext("weights"),
                            options,
                            cache);
  }

  return *weights;
}

popnn::gru::GruParams GRUOpx::createGRUParams(const GRUOp &gru_op) {
  auto in_info         = gru_op.inInfo(GRUOp::getInputInIndex());
  auto seq_length      = static_cast<unsigned>(gru_op.getSeqLength());
  auto batch_size      = static_cast<unsigned>(gru_op.getBatchSize());
  unsigned input_size  = static_cast<unsigned>(gru_op.getInputSize());
  unsigned hidden_size = static_cast<unsigned>(gru_op.getHiddenSize());

  auto params = popnn::gru::GruParams(
      popType(in_info), batch_size, seq_length, {input_size, hidden_size});
  params.resetAfter = gru_op.getLinearBeforeResetAttribute() != 0;
  return params;
}

popnn::gru::GruParams GRUOpx::createGRUParams() const {
  auto &gru_op = getOp<GRUOp>();
  return createGRUParams(gru_op);
}

std::set<TensorId> GRUOpx::mustExistBeforeCreate(InIndex) const { return {}; }

void GRUOpx::prepareWeights(poplar::program::Sequence &prog) const {
  // check to see if the weights were created
  prog.add(poplar::program::Copy(
      getInTensor(GRUOp::getWeightsInIndex()),
      reshapePoplibWeightsForOnnx(getGRUWeights().inputWeights),
      false,
      debugContext()));
  prog.add(poplar::program::Copy(
      getInTensor(GRUOp::getRecurrenceInIndex()),
      reshapePoplibWeightsForOnnx(getGRUWeights().outputWeights),
      false,
      debugContext()));
}

poplar::Tensor GRUOpx::getInput(poplar::program::Sequence &prog) const {
  if (!inputCreated(GRUOp::getInputInIndex())) {
    auto input =
        createInputTensor(GRUOp::getInputInIndex(), getDebugNameAndId("input"))
            .getPoplarTensor();
    auto raw_input = getInTensor(GRUOp::getInputInIndex());
    prog.add(poplar::program::Copy(raw_input, input, false, debugContext()));
    return input;
  } else {
    return getInTensor(GRUOp::getInputInIndex());
  }
}

GRUGradOpx::GRUGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<GRUGradOp>(op, Onnx::GradOperators::GRUGrad);
}

void GRUGradOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor init_state_h =
      getInTensor(GRUGradOp::getInitStateOutputInIndex());

  popnn::gru::GruWeights weights;
  weights.inputWeights  = getInTensor(GRUGradOp::getInputWeightsInIndex());
  weights.outputWeights = getInTensor(GRUGradOp::getOutputWeightsInIndex());
  weights.biases        = getInTensor(GRUGradOp::getBiasesInIndex());

  auto intermediates  = getInTensor(GRUGradOp::getIntermediatesInIndex());
  auto forward_input  = getInTensor(GRUGradOp::getInputInIndex());
  auto forward_output = getInTensor(GRUGradOp::getOutputInIndex());

  auto &gru_grad_op   = getOp<GRUGradOp>();
  auto &gru_op        = gru_grad_op.getForwardOp();
  auto batch_size     = static_cast<unsigned>(gru_op.getBatchSize());
  auto hidden_size    = static_cast<unsigned>(gru_op.getHiddenSize());
  auto seq_length     = static_cast<unsigned>(gru_op.getSeqLength());
  auto num_directions = static_cast<unsigned>(gru_op.getNumDirections());
  auto gru_params     = createGRUParams();

  auto output_grad = getInTensor(GRUGradOp::getOutputGradInIndex())
                         .reshape({seq_length, batch_size, hidden_size});

  output_grad = cloneNcopy(prog, output_grad);

  auto output_h_grad = getHiddenStateGrad();

  if (gru_op.getLinearBeforeResetAttribute()) {
    // poplibs shapes the gradient tensors just like the corresponding
    // weights - need to ensure correct shape here
    weights.biases = weights.biases.reshape({3, 2, hidden_size});
  }

  popops::addInPlace(graph().getPoplarGraph(),
                     output_grad[output_grad.dim(0) - 1],
                     output_h_grad,
                     prog,
                     debugContext());

  poplar::Tensor input_grad;
  popnn::gru::GruWeights weights_grad;

  auto init_state_grad = gruBwdWithWU(graph().getPoplarGraph(),
                                      gru_params,
                                      prog,
                                      init_state_h,
                                      intermediates,
                                      weights,
                                      forward_input,
                                      forward_output,
                                      output_grad,
                                      &input_grad,
                                      weights_grad,
                                      debugContext("gruBwdWithWU"),
                                      dv_p->lowering().lstmOptions,
                                      &dv_p->matmulCache);

  setOutTensor(GRUGradOp::getInputOutIndex(), input_grad);
  setOutTensor(GRUGradOp::getWeightsOutIndex(),
               GRUOpx::reshapePoplibWeightsForOnnx(weights_grad.inputWeights));
  setOutTensor(GRUGradOp::getRecurrenceOutIndex(),
               GRUOpx::reshapePoplibWeightsForOnnx(weights_grad.outputWeights));

  if (gru_op.hasBiasInput()) {
    auto b_grad = GRUOpx::reshapePoplibBiasesForOnnx(weights_grad.biases);

    if (gru_op.getLinearBeforeResetAttribute()) {
      // separate gradients for input and hidden bias
      b_grad = b_grad.reshape({1, 6 * hidden_size});
      setOutTensor(GRUGradOp::getBiasOutIndex(), b_grad);
    } else {
      // propagate same gradient to both input and hidden bias
      setOutTensor(GRUGradOp::getBiasOutIndex(),
                   poplar::concat({b_grad, b_grad}, 1));
    }
  }
  if (gru_op.hasInitialHInput()) {
    setOutTensor(
        GRUGradOp::getInitialHOutIndex(),
        init_state_grad.reshape({num_directions, batch_size, hidden_size}));
  }
}

poplar::Tensor GRUGradOpx::getHiddenStateGrad() const {
  auto &gru_grad_op = getOp<GRUGradOp>();
  auto &gru_op      = gru_grad_op.getForwardOp();

  unsigned batch_size  = static_cast<unsigned>(gru_op.getBatchSize());
  unsigned hidden_size = static_cast<unsigned>(gru_op.getHiddenSize());

  auto elem_type = getInTensor(GRUGradOp::getOutputGradInIndex()).elementType();

  if (gru_grad_op.hasHiddenStateGradInput()) {
    return getInTensor(GRUGradOp::getHiddenStateOutputGradInIndex())
        .reshape({batch_size, hidden_size});
  } else {
    auto zero = getScalarVariable(elem_type, "gru/zero_hidden_state");
    graph().getPoplarGraph().setTileMapping(zero, 0);
    graph().getPoplarGraph().setInitialValue(zero, 0);
    zero = zero.expand({0, 0});
    zero = zero.broadcast(batch_size, 0);
    zero = zero.broadcast(hidden_size, 1);
    return zero;
  }
}

popnn::gru::GruParams GRUGradOpx::createGRUParams() const {
  auto &gru_grad_op = getOp<GRUGradOp>();
  return GRUOpx::createGRUParams(gru_grad_op.getForwardOp());
}

namespace {
OpxCreator<GRUOpx> gruOpxCreator({Onnx::Operators::GRU_7});
OpxCreator<GRUGradOpx> gruGradOpxCreator(Onnx::GradOperators::GRUGrad);
} // namespace

} // namespace popx
} // namespace popart
