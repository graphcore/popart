// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/op/exp.hpp>
#include <popart/op/mul.hpp>
#include <popart/patterns/expgradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool ExpGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<ExpGradOp>();
}

std::vector<const Tensor *> ExpGradOpPattern::touches(Op *) const { return {}; }

// grad_out = grad_in * fwd_out
bool ExpGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(ExpGradOp::getGradInIndex());
  auto fwd_out  = op->inTensor(ExpGradOp::getFwdOutInIndex());
  auto grad_out = op->outTensor(ExpGradOp::getOutIndex());

  // create the new ops
  auto mul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Remove the ExpGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(), fwd_out->id);
  mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<ExpGradOpPattern>
    ExpGradOpPattern(PreAliasPatternType::ExpGradOp, "ExpGradOp");
}

} // namespace popart
