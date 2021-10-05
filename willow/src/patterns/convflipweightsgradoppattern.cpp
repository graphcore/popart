// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/convflipweightsgradoppattern.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool ConvFlipWeightsGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<ConvFlipWeightsGradOp>();
}

std::vector<const Tensor *> ConvFlipWeightsGradOpPattern::touches(Op *) const {
  return {};
}

// Turn the op into a convflip
bool ConvFlipWeightsGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(ConvFlipWeightsOp::getInIndex());
  auto grad_out = op->outTensor(ConvFlipWeightsOp::getOutIndex());

  // Create the new op and transfer properties
  auto convflip = dynamic_cast<ConvFlipWeightsOp *>(
      makeReplacementOpInIr(Onnx::CustomOperators::ConvFlipWeights, op));
  auto gradOp = dynamic_cast<ConvFlipWeightsGradOp *>(op);
  convflip->setGroupReshape(gradOp->getGroupReshape());
  convflip->setParameters(gradOp->getParameters());
  convflip->setConvOptions(gradOp->getMultiConvOptions());

  // Remove the ConvFlipWeightsGradOp op
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect to the conv flip op
  convflip->connectInTensor(ConvFlipWeightsOp::getInIndex(), grad_in->id);
  convflip->connectOutTensor(ConvFlipWeightsOp::getOutIndex(), grad_out->id);
  convflip->setup();
  return true;
}

namespace {
static PatternCreator<ConvFlipWeightsGradOpPattern>
    ConvFlipWeightsGradOpPattern("ConvFlipWeightsGradOp", true, true);
}

} // namespace popart
