// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/Lstm.hpp>
#include <popart/error.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/popartlstmx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/optional.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {

namespace {

poplar::Tensor concatWeights(const poplar::Tensor &inputWeights,
                             const poplar::Tensor &outputWeights) {
  return poplar::concat(inputWeights, outputWeights, 1);
}

poplar::OptionFlags
addAvailableMemoryProportionOption(const PopartLSTMOp &op,
                                   const poplar::OptionFlags &flags) {
  auto rflags     = flags;
  const auto &amp = op.getAvailableMemoryProportion();
  if (amp.has_value()) {
    rflags.set("availableMemoryProportion", std::to_string(*amp));
  }
  return rflags;
}

} // namespace

PopartLSTMOpx::PopartLSTMOpx(Op *op, Devicex *devicex)
    : PopartLSTMOpxBase(op, devicex) {
  verifyOp<PopartLSTMOp>(op, Onnx::CustomOperators::LSTM_1);
}

void PopartLSTMOpx::grow(poplar::program::Sequence &prog) const {
  auto input         = getInTensor(PopartLSTMOp::getInputInIndex());
  auto &lstm_op      = getOp<PopartLSTMOp>();
  auto seq_lens      = getSeqLens();
  auto lstmWeights   = getWeights(prog);
  auto initState     = getInitialState(prog);
  auto intermediates = getIntermediates();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);

  auto params = createLSTMParams(lstm_op, seq_lens);
  poplar::Tensor output;
  poplar::Tensor cellState;
  std::tie(output, cellState) = popnn::lstm::lstmFwd(graph(),
                                                     params,
                                                     initState,
                                                     input,
                                                     lstmWeights,
                                                     intermediates.get(),
                                                     prog,
                                                     debugContext("lstmFwd"),
                                                     options,
                                                     &dv_p->matmulCache);

  setOutTensor(PopartLSTMOp::getOutputOutIndex(), output);
  setOutTensor(PopartLSTMOp::getCellStateOutIndex(), cellState);

  if (hasOutput(PopartLSTMOp::getIntermediatesOutIndex())) {
    setOutTensor(PopartLSTMOp::getIntermediatesOutIndex(), *intermediates);
  }
}

std::unique_ptr<poplar::Tensor> PopartLSTMOpx::getIntermediates() const {
  std::unique_ptr<poplar::Tensor> intermediates = nullptr;
  if (hasOutput(PopartLSTMOp::getIntermediatesOutIndex())) {
    intermediates = std::make_unique<poplar::Tensor>();
  }
  return intermediates;
}

InputCreatorType PopartLSTMOpx::getInputCreatorType(InIndex index) const {
  if (index == PopartLSTMOp::getInputInIndex() ||
      index == PopartLSTMOp::getWeightsInIndex() ||
      index == PopartLSTMOp::getBiasesInIndex() ||
      index == PopartLSTMOp::getInitialStateInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

poplar::Tensor
PopartLSTMOpx::createInput(InIndex index,
                           const poplar::DebugNameAndId &) const {
  if (index == PopartLSTMOp::getInputInIndex()) {
    return createLSTMInput();
  } else if (index == PopartLSTMOp::getWeightsInIndex()) {
    return createWeightsInput();
  } else if (index == PopartLSTMOp::getBiasesInIndex()) {
    return createBiasesInput();
  } else if (index == PopartLSTMOp::getInitialStateInIndex()) {
    return createInitialStateInput().getAsTensor();
  } else {
    throw error("PopartLSTMOpx::createInput is not supported for index {}",
                index);
  }
}

poplar::Tensor PopartLSTMOpx::createLSTMInput() const {
  auto &lstm_op = getOp<PopartLSTMOp>();
  auto seq_lens = getSeqLens();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);
  return popnn::lstm::createInput(graph(),
                                  createLSTMParams(lstm_op, seq_lens),
                                  getDebugNameAndId("createLSTMInput"),
                                  options,
                                  &dv_p->matmulCache);
}

poplar::Tensor PopartLSTMOpx::createWeightsInput() const {
  poplar::Tensor inputWeights, outputWeights;
  auto &lstm_op = getOp<PopartLSTMOp>();
  auto seq_lens = getSeqLens();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);
  std::tie(inputWeights, outputWeights) =
      popnn::lstm::createWeightsKernel(graph(),
                                       createLSTMParams(lstm_op, seq_lens),
                                       debugContext("weights"),
                                       options,
                                       &dv_p->matmulCache);
  return concatWeights(inputWeights, outputWeights);
}

std::set<TensorId> PopartLSTMOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

PopartLSTMGradOpx::PopartLSTMGradOpx(Op *op, Devicex *devicex)
    : PopartLSTMOpxBase(op, devicex) {
  verifyOp<PopartLSTMGradOp>(op, Onnx::GradOperators::PopartLSTMGrad);
}

void PopartLSTMGradOpx::grow(poplar::program::Sequence &prog) const {
  auto intermediates = getInTensor(PopartLSTMGradOp::getIntermediatesInIndex());
  auto forwardInput  = getInTensor(PopartLSTMGradOp::getInputInIndex());
  auto forwardOutput = getInTensor(PopartLSTMGradOp::getFwdOutputInIndex());
  auto forwardOutputGrad =
      getInTensor(PopartLSTMGradOp::getFwdOutputGradInIndex());

  popnn::lstm::LstmState lastStepStateGrad;
  popnn::lstm::LstmState *lastStepStateGradPtr = nullptr;
  if (hasInput(PopartLSTMGradOp::getFwdCellStateGradInIndex())) {
    lastStepStateGrad.cellState =
        getInTensor(PopartLSTMGradOp::getFwdCellStateGradInIndex());
    lastStepStateGradPtr = &lastStepStateGrad;
  }

  auto initState   = getInitialState(prog);
  auto lstmWeights = getWeights(prog);

  poplar::Tensor inputGrad;
  popnn::lstm::LstmWeights weightsGrad;
  auto &grad_lstm_op = getOp<PopartLSTMGradOp>();
  auto seq_lens      = getSeqLens();
  auto params        = createLSTMParams(grad_lstm_op, seq_lens);
  auto initStateGrad = lstmBwdWithWU(graph(),
                                     params,
                                     prog,
                                     initState,
                                     intermediates,
                                     lstmWeights,
                                     forwardInput,
                                     forwardOutput,
                                     forwardOutputGrad,
                                     lastStepStateGradPtr,
                                     &inputGrad,
                                     weightsGrad,
                                     debugContext("lstmBwdWithWU"),
                                     dv_p->lowering().lstmOptions,
                                     &dv_p->matmulCache);

  auto weightsOut =
      concatWeights(weightsGrad.inputWeights, weightsGrad.outputWeights);

  setOutTensor(PopartLSTMGradOp::getInputOutIndex(), inputGrad);
  setOutTensor(PopartLSTMGradOp::getWeightsOutIndex(), weightsOut);
  if (hasOutput(PopartLSTMGradOp::getBiasesOutIndex())) {
    setOutTensor(PopartLSTMGradOp::getBiasesOutIndex(), weightsGrad.biases);
  }
  if (hasOutput(PopartLSTMGradOp::getInitialStateOutIndex())) {
    setOutTensor(PopartLSTMGradOp::getInitialStateOutIndex(),
                 initStateGrad.getAsTensor());
  }
}

namespace {
OpxCreator<PopartLSTMOpx> popartLstmOpxCreator(Onnx::CustomOperators::LSTM_1);
OpxCreator<PopartLSTMGradOpx>
    popartLstmGradOpxCreator(Onnx::GradOperators::PopartLSTMGrad);
} // namespace

} // namespace popx
} // namespace popart
