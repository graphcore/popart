// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <tuple>
#include <poplar/OptionFlags.hpp>
#include <poplar/Tensor.hpp>
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
#include "popart/popx/popopx.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/optional.hpp"

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

void PopartLSTMOpx::grow(snap::program::Sequence &prog) const {
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
  std::tie(output, cellState) = popnn::lstm::lstmFwd(graph().getPoplarGraph(),
                                                     params,
                                                     initState,
                                                     input.getPoplarTensor(),
                                                     lstmWeights,
                                                     intermediates.get(),
                                                     prog.getPoplarSequence(),
                                                     debugContext("lstmFwd"),
                                                     options,
                                                     &dv_p->matmulCache);

  setOutTensor(PopartLSTMOp::getOutputOutIndex(),
               snap::Tensor{output, graph()});
  setOutTensor(PopartLSTMOp::getCellStateOutIndex(),
               snap::Tensor{cellState, graph()});

  if (hasOutput(PopartLSTMOp::getIntermediatesOutIndex())) {
    setOutTensor(PopartLSTMOp::getIntermediatesOutIndex(),
                 snap::Tensor{*intermediates, graph()});
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

snap::Tensor
PopartLSTMOpx::createInputTensor(InIndex index,
                                 const poplar::DebugNameAndId &) const {
  if (index == PopartLSTMOp::getInputInIndex()) {
    return createLSTMInput();
  } else if (index == PopartLSTMOp::getWeightsInIndex()) {
    return createWeightsInput();
  } else if (index == PopartLSTMOp::getBiasesInIndex()) {
    return createBiasesInput();
  } else if (index == PopartLSTMOp::getInitialStateInIndex()) {
    return snap::Tensor{createInitialStateInput().getAsTensor(), graph()};
  } else {
    throw error("PopartLSTMOpx::createInput is not supported for index {}",
                index);
  }
}

snap::Tensor PopartLSTMOpx::createLSTMInput() const {
  auto &lstm_op = getOp<PopartLSTMOp>();
  auto seq_lens = getSeqLens();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);
  return snap::Tensor{
      popnn::lstm::createInput(graph().getPoplarGraph(),
                               createLSTMParams(lstm_op, seq_lens),
                               getDebugNameAndId("createLSTMInput"),
                               options,
                               &dv_p->matmulCache),
      graph()};
}

snap::Tensor PopartLSTMOpx::createWeightsInput() const {
  poplar::Tensor inputWeights, outputWeights;
  auto &lstm_op = getOp<PopartLSTMOp>();
  auto seq_lens = getSeqLens();
  auto options =
      addAvailableMemoryProportionOption(lstm_op, dv_p->lowering().lstmOptions);
  std::tie(inputWeights, outputWeights) =
      popnn::lstm::createWeightsKernel(graph().getPoplarGraph(),
                                       createLSTMParams(lstm_op, seq_lens),
                                       debugContext("weights"),
                                       options,
                                       &dv_p->matmulCache);
  return snap::Tensor{concatWeights(inputWeights, outputWeights), graph()};
}

std::set<TensorId> PopartLSTMOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

PopartLSTMGradOpx::PopartLSTMGradOpx(Op *op, Devicex *devicex)
    : PopartLSTMOpxBase(op, devicex) {
  verifyOp<PopartLSTMGradOp>(op, Onnx::GradOperators::PopartLSTMGrad);
}

void PopartLSTMGradOpx::grow(snap::program::Sequence &prog) const {
  auto intermediates = getInTensor(PopartLSTMGradOp::getIntermediatesInIndex());
  auto forwardInput  = getInTensor(PopartLSTMGradOp::getInputInIndex());
  auto forwardOutput = getInTensor(PopartLSTMGradOp::getFwdOutputInIndex());
  auto forwardOutputGrad =
      getInTensor(PopartLSTMGradOp::getFwdOutputGradInIndex());

  popnn::lstm::LstmState lastStepStateGrad;
  popnn::lstm::LstmState *lastStepStateGradPtr = nullptr;
  if (hasInput(PopartLSTMGradOp::getFwdCellStateGradInIndex())) {
    lastStepStateGrad.cellState =
        getInTensor(PopartLSTMGradOp::getFwdCellStateGradInIndex())
            .getPoplarTensor();
    lastStepStateGradPtr = &lastStepStateGrad;
  }

  auto initState   = getInitialState(prog);
  auto lstmWeights = getWeights(prog);

  poplar::Tensor inputGrad;
  popnn::lstm::LstmWeights weightsGrad;
  auto &grad_lstm_op = getOp<PopartLSTMGradOp>();
  auto seq_lens      = getSeqLens();
  auto params        = createLSTMParams(grad_lstm_op, seq_lens);
  auto initStateGrad = lstmBwdWithWU(graph().getPoplarGraph(),
                                     params,
                                     prog.getPoplarSequence(),
                                     initState,
                                     intermediates.getPoplarTensor(),
                                     lstmWeights,
                                     forwardInput.getPoplarTensor(),
                                     forwardOutput.getPoplarTensor(),
                                     forwardOutputGrad.getPoplarTensor(),
                                     lastStepStateGradPtr,
                                     &inputGrad,
                                     weightsGrad,
                                     debugContext("lstmBwdWithWU"),
                                     dv_p->lowering().lstmOptions,
                                     &dv_p->matmulCache);

  auto weightsOut =
      concatWeights(weightsGrad.inputWeights, weightsGrad.outputWeights);

  setOutTensor(PopartLSTMGradOp::getInputOutIndex(),
               snap::Tensor{inputGrad, graph()});
  setOutTensor(PopartLSTMGradOp::getWeightsOutIndex(),
               snap::Tensor{weightsOut, graph()});
  if (hasOutput(PopartLSTMGradOp::getBiasesOutIndex())) {
    setOutTensor(PopartLSTMGradOp::getBiasesOutIndex(),
                 snap::Tensor{weightsGrad.biases, graph()});
  }
  if (hasOutput(PopartLSTMGradOp::getInitialStateOutIndex())) {
    setOutTensor(PopartLSTMGradOp::getInitialStateOutIndex(),
                 snap::Tensor{initStateGrad.getAsTensor(), graph()});
  }
}

namespace {
OpxCreator<PopartLSTMOpx> popartLstmOpxCreator(Onnx::CustomOperators::LSTM_1);
OpxCreator<PopartLSTMGradOpx>
    popartLstmGradOpxCreator(Onnx::GradOperators::PopartLSTMGrad);
} // namespace

} // namespace popx
} // namespace popart
