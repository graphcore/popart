// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/div.hpp>
#include <popart/op/log.hpp>
#include <popart/patterns/loggradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool LogGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<LogGradOp>();
}

std::vector<const Tensor *> LogGradOpPattern::touches(Op *) const { return {}; }

// grad_out = grad_in / fwd_in
bool LogGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(LogGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(LogGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(LogGradOp::getOutIndex());

  // create the new ops
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);

  // Remove the LogGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(), fwd_in->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);
  div->setup();

  return true;
}

namespace {
static PatternCreator<LogGradOpPattern>
    LogGradOpPattern(PreAliasPatternType::LOGGRADOP, "LogGradOp");
}

} // namespace popart
