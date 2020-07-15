// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/addbias.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convdatagrad.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

namespace popart {

bool ConvDataGradPattern::matches(Op *op) const {
  return (op->opid == Onnx::GradOperators::ConvDataGrad);
}

std::vector<const Tensor *> ConvDataGradPattern::touches(Op *) const {
  return {};
}

bool ConvDataGradPattern::apply(Op *op) const {

  auto weights_in     = op->inTensor(ConvDataGradOp::getWeightsInIndex());
  auto gradConvIn_out = op->inTensor(ConvDataGradOp::getGradConvolvedInIndex());
  auto grad_out       = op->outTensor(ConvDataGradOp::getOutIndex());

  const auto gradOp = dynamic_cast<ConvDataGradOp *>(op);

  auto flip = dynamic_cast<ConvFlipWeightsOp *>(
      makeReplacementOpInIr(Onnx::CustomOperators::ConvFlipWeights, op));
  auto conv = dynamic_cast<ConvOp *>(
      makeReplacementOpInIr(Onnx::Operators::Conv_1, op));

  // Inherit the options from the forward op
  flip->setConvOptions(gradOp->getConvOptions());
  conv->setConvOptions(gradOp->getConvOptions());

  gradOp->disconnectAllInputs();
  gradOp->disconnectAllOutputs();

  // Configure the flip weight op
  flip->connectInTensor(ConvFlipWeightsOp::getInIndex(), weights_in->id);
  flip->createAndConnectOutTensor(
      ConvFlipWeightsOp::getOutIndex(),
      weights_in->getIr().createIntermediateTensorId(weights_in->id));
  flip->setup();
  flip->setParameters(gradOp->getParameters());

  // Configure the conv op for the bwd pass
  conv->connectInTensor(ConvOp::getWeightsInIndex(),
                        flip->outTensor(ConvFlipWeightsOp::getOutIndex())->id);
  conv->connectInTensor(ConvOp::getDataInIndex(), gradConvIn_out->id);
  conv->connectOutTensor(ConvOp::getOutIndex(), grad_out->id);
  conv->setupFromDataGradOp(gradOp);

  // Remove the ConvGradOp
  gradOp->getGraph().eraseOp(gradOp->id);

  return true;
}

namespace {
static PatternCreator<ConvDataGradPattern>
    convDataGradPattern(PreAliasPatternType::ConvDataGrad, "ConvDataGrad");
}

} // namespace popart
