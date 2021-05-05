// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/lstm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/popartlstmx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <popnn/Lstm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

namespace {

poplar::Tensor concatWeights(const poplar::Tensor &inputWeights,
                             const poplar::Tensor &outputWeights) {
  return poplar::concat(inputWeights, outputWeights, 1);
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

  auto params = createLSTMParams(lstm_op, seq_lens);
  poplar::Tensor output;
  poplar::Tensor cellState;
  std::tie(output, cellState) =
      popnn::lstm::lstmFwd(graph(),
                           params,
                           initState,
                           input,
                           lstmWeights,
                           intermediates.get(),
                           prog,
                           debugContext("lstmFwd"),
                           dv_p->lowering().lstmOptions,
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
  return popnn::lstm::createInput(graph(),
                                  createLSTMParams(lstm_op, seq_lens),
                                  getDebugNameAndId("createLSTMInput"),
                                  dv_p->lowering().lstmOptions,
                                  &dv_p->matmulCache);
}

poplar::Tensor PopartLSTMOpx::createWeightsInput() const {
  poplar::Tensor inputWeights, outputWeights;
  auto &lstm_op = getOp<PopartLSTMOp>();
  auto seq_lens = getSeqLens();
  std::tie(inputWeights, outputWeights) =
      popnn::lstm::createWeightsKernel(graph(),
                                       createLSTMParams(lstm_op, seq_lens),
                                       debugContext("weights"),
                                       dv_p->lowering().lstmOptions,
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

  const poplar::Tensor *forwardCellStateGrad = nullptr;
  if (hasInput(PopartLSTMGradOp::getFwdCellStateGradInIndex())) {
    forwardCellStateGrad =
        &getInTensor(PopartLSTMGradOp::getFwdCellStateGradInIndex());
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
                                     forwardCellStateGrad,
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
