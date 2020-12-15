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

GRUOpx::GRUOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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
        popnn::gru::createInitialState(graph(),
                                       createGRUParams(),
                                       debugPrefix("initialState"),
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
    popops::zero(graph(), init_state_h, prog, debugPrefix());
  }

  // Check the inputs have been created
  if (hasInitH) {
    prog.add(
        poplar::program::Copy(getInTensor(GRUOp::getInitialHInIndex()),
                              createInput(GRUOp::getInitialHInIndex(), "initH"),
                              false,
                              debugPrefix()));
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
    output = popnn::gru::gruFwd(graph(),
                                createGRUParams(),
                                init_state_h,
                                input,
                                *weights,
                                intermediate.get(),
                                prog,
                                debugPrefix("gruFwd"),
                                dv_p->lowering().lstmOptions,
                                &dv_p->matmulCache);
  } else if (gru_op.getDirectionAttribute() == "backward") {
    output = popnn::gru::gruFwd(graph(),
                                createGRUParams(),
                                init_state_h,
                                input.reverse(0),
                                *weights,
                                intermediate.get(),
                                prog,
                                debugPrefix("gruFwd"),
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

  auto output_h_state = output[createGRUParams().timeSteps - 1];

  // cloneNcopy to ensure outputs are not aliases of each other
  // TODO T18126 remove requirement for this cloneNcopy
  reshapeAndInsert(GRUOp::getHiddenStateOutIndex(),
                   cloneNcopy(prog, output_h_state));

  setOutTensor(GRUOp::getInitStateOutputPassThroughIndex(), init_state_h);

  setOutTensor(GRUOp::getInputWeightsPassThroughIndex(), weights->inputWeights);
  setOutTensor(GRUOp::getOutputWeightsPassThroughIndex(),
               weights->outputWeights);
  setOutTensor(GRUOp::getBiasesPassThroughIndex(), weights->biases);

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
  // bias in onnx is shape [num_directions, 6 * hidden_size]
  // bias in poplibs is [3, hidden_size]
  auto &gru_op         = getOp<GRUOp>();
  unsigned hidden_size = static_cast<unsigned>(gru_op.getHiddenSize());

  auto biases = reshapePoplibWeightsForOnnx(getGRUWeights().biases, false);

  if (gru_op.hasBiasInput()) {
    auto bias_input = getInTensor(GRUOp::getBiasInIndex());

    poplar::program::Copy copyProg(
        bias_input.slice(0, 3 * hidden_size, 1), biases, false, debugPrefix());
    prog.add(copyProg);

    popops::mapInPlace(graph(),
                       popops::expr::BinaryOpType::ADD,
                       biases,
                       bias_input.slice(3 * hidden_size, 6 * hidden_size, 1),
                       prog,
                       debugPrefix("add"));
  } else {
    popops::zero(graph(), biases, prog, debugPrefix("zero"));
  }
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

poplar::Tensor GRUOpx::createInput(InIndex index, const std::string &) const {
  createdInputs.insert(index);

  if (index == GRUOp::getInputInIndex()) {
    return createGRUInput();
  } else if (index == GRUOp::getWeightsInIndex()) {
    auto inputWeights = getGRUWeights().inputWeights;
    return reshapePoplibWeightsForOnnx(inputWeights, true);
  } else if (index == GRUOp::getRecurrenceInIndex()) {
    auto outputWeights = getGRUWeights().outputWeights;
    return reshapePoplibWeightsForOnnx(outputWeights, true);
  } else if (index == GRUOp::getInitialHInIndex()) {
    auto &gru_op = getOp<GRUOp>();

    unsigned batch_size     = static_cast<unsigned>(gru_op.getBatchSize());
    unsigned hidden_size    = static_cast<unsigned>(gru_op.getHiddenSize());
    unsigned num_directions = static_cast<unsigned>(gru_op.getNumDirections());

    auto init_h = getInitialState();
    return init_h.reshape({num_directions, batch_size, hidden_size});
  } else {
    throw error("GRUOpx::createInput is not supported for index {}", index);
  }
}

bool GRUOpx::inputCreated(InIndex index) const {
  return createdInputs.count(index) > 0;
}

poplar::Tensor
GRUOpx::reshapePoplibWeightsForOnnx(poplar::Tensor poplib_weights,
                                    bool transpose) {
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

  if (transpose) {
    for (int i = 0; i < slices.size(); i++) {
      slices[i] = slices[i].dimShuffle({0, 2, 1});
    }
  }

  auto wz = slices[0];
  auto wr = slices[1];
  auto wh = slices[2];

  return poplar::concat({wr, wz, wh}, 1);
}

poplar::Tensor GRUOpx::createGRUInput() const {
  auto gru_params = createGRUParams();
  auto options    = dv_p->lowering().lstmOptions;
  auto cache      = &dv_p->matmulCache;

  return popnn::gru::createInput(
      graph(), gru_params, debugPrefix("input"), options, cache);
}

popnn::gru::GruWeights GRUOpx::getGRUWeights() const {
  if (!weights) {
    auto gru_params = createGRUParams();
    auto options    = dv_p->lowering().lstmOptions;
    auto cache      = &dv_p->matmulCache;

    weights = createWeights(
        graph(), gru_params, debugPrefix("weights"), options, cache);
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
  return params;
}

popnn::gru::GruParams GRUOpx::createGRUParams() const {
  auto &gru_op = getOp<GRUOp>();
  return createGRUParams(gru_op);
}

std::vector<TensorId> GRUOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

void GRUOpx::prepareWeights(poplar::program::Sequence &prog) const {
  // check to see if the weights were created
  prog.add(poplar::program::Copy(
      getInTensor(GRUOp::getWeightsInIndex()),
      reshapePoplibWeightsForOnnx(getGRUWeights().inputWeights, true),
      false,
      debugPrefix()));
  prog.add(poplar::program::Copy(
      getInTensor(GRUOp::getRecurrenceInIndex()),
      reshapePoplibWeightsForOnnx(getGRUWeights().outputWeights, true),
      false,
      debugPrefix()));
}

poplar::Tensor GRUOpx::getInput(poplar::program::Sequence &prog) const {
  if (!inputCreated(GRUOp::getInputInIndex())) {
    auto input     = createInput(GRUOp::getInputInIndex(), "input");
    auto raw_input = getInTensor(GRUOp::getInputInIndex());
    prog.add(poplar::program::Copy(raw_input, input, false, debugPrefix()));
    return input;
  } else {
    return getInTensor(GRUOp::getInputInIndex());
  }
}

GRUGradOpx::GRUGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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

  popops::addInPlace(graph(),
                     output_grad[output_grad.dim(0) - 1],
                     output_h_grad,
                     prog,
                     debugPrefix());

  poplar::Tensor input_grad;
  popnn::gru::GruWeights weights_grad;

  auto init_state_grad = gruBwdWithWU(graph(),
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
                                      debugPrefix("gruBwdWithWU"),
                                      dv_p->lowering().lstmOptions,
                                      &dv_p->matmulCache);

  setOutTensor(GRUGradOp::getInputOutIndex(), input_grad);
  setOutTensor(
      GRUGradOp::getWeightsOutIndex(),
      GRUOpx::reshapePoplibWeightsForOnnx(weights_grad.inputWeights, true));
  setOutTensor(
      GRUGradOp::getRecurrenceOutIndex(),
      GRUOpx::reshapePoplibWeightsForOnnx(weights_grad.outputWeights, true));

  if (gru_op.hasBiasInput()) {
    auto b_grad =
        GRUOpx::reshapePoplibWeightsForOnnx(weights_grad.biases, false);
    setOutTensor(GRUGradOp::getBiasOutIndex(),
                 poplar::concat({b_grad, b_grad}, 1));
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
    graph().setTileMapping(zero, 0);
    graph().setInitialValue(zero, 0);
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
