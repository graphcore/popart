// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/div.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/patterns/sqrtgradoppattern.hpp>
#include <popart/tensor.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

namespace popart {

bool SqrtGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SqrtGradOp>();
}

std::vector<const Tensor *> SqrtGradOpPattern::touches(Op *) const {
  return {};
}

bool SqrtGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(SqrtGradOp::getGradInIndex());
  auto fwd_out  = op->inTensor(SqrtGradOp::getFwdOutInIndex());
  auto grad_out = op->outTensor(SqrtGradOp::getOutIndex());

  // create the new ops
  auto scale = dynamic_cast<ScaleOp *>(
      makeReplacementOpInIr(Onnx::AiGraphcore::OpSet1::Scale, op));
  scale->setScaleFactor(2.0f);
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Remove the DivArg0GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  scale->connectInTensor(ScaleOp::getInIndex(), fwd_out->id);
  scale->createAndConnectOutTensor(
      ScaleOp::getOutIndex(),
      fwd_out->getIr().createIntermediateTensorId(fwd_out->id));
  scale->setup();

  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(),
                       scale->outTensor(ScaleOp::getOutIndex())->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<SqrtGradOpPattern>
    SqrtGradOpPattern("SqrtGradOp", true, true);
}

} // namespace popart
