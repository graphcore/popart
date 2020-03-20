// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cos.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/sin.hpp>
#include <popart/patterns/cosgradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool CosGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<CosGradOp>();
}

std::vector<const Tensor *> CosGradOpPattern::touches(Op *) const { return {}; }

// grad_out = - grad_in * sin(fwd_in)
bool CosGradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(CosGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(CosGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(CosGradOp::getOutIndex());

  // create the new ops
  auto sin    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sin, op);
  auto mul    = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto negate = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Neg, op);

  // Remove the CosGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  sin->connectInTensor(SinOp::getInIndex(), fwd_in->id);
  sin->createAndConnectOutTensor(
      SinOp::getOutIndex(),
      grad_in->getIr().createIntermediateTensorId(grad_in->id));
  sin->setup();

  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(),
                       sin->outTensor(SinOp::getOutIndex())->id);
  mul->createAndConnectOutTensor(
      MulOp::getOutIndex(),
      grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul->setup();

  negate->connectInTensor(NegateOp::getInIndex(),
                          mul->outTensor(MulOp::getOutIndex())->id);
  negate->connectOutTensor(NegateOp::getOutIndex(), grad_out->id);

  return true;
}

namespace {
static PatternCreator<CosGradOpPattern>
    CosGradOpPattern(PreAliasPatternType::COSGRADOP, "CosGradOp");
}

} // namespace popart
