// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/log.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/pow.hpp>
#include <popart/patterns/powarg1gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool PowArg1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<PowArg1GradOp>();
}

// grad_out = grad_in * out * log(arg_0)
TensorId PowArg1GradOpPattern::makeAllReplacementOps(Op *op,
                                                     Tensor *grad_in,
                                                     Tensor *fwd_in0,
                                                     Tensor *fwd_in1,
                                                     Tensor *fwd_out) const {
  // create the new ops
  auto log   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);
  auto mul_1 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul_2 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Connect up the new ops
  log->connectInTensor(0, fwd_in0->id);
  log->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  log->outInfo(0) = log->inInfo(0);

  mul_1->connectInTensor(0, fwd_out->id);
  mul_1->connectInTensor(1, log->outTensor(0)->id);
  mul_1->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul_1->outInfo(0) = op->prettyNpOut(mul_1->inInfo(0), mul_1->inInfo(1));

  mul_2->connectInTensor(0, grad_in->id);
  mul_2->connectInTensor(1, mul_1->outTensor(0)->id);
  mul_2->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  mul_2->outInfo(0) = op->prettyNpOut(mul_2->inInfo(0), mul_2->inInfo(1));

  return mul_2->outTensor(0)->id;
}

namespace {
static PatternCreator<popart::PowArg1GradOpPattern>
    PowArg1GradOpPattern(PreAliasPatternType::PowArg1GradOp,
                         "PowArg1GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
