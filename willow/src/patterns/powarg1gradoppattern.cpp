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
TensorId
PowArg1GradOpPattern::makeAllReplacementOps(Op *op,
                                            Ir *ir,
                                            const Tensor &gradIn,
                                            const Tensor &fwdIn0,
                                            const Tensor &fwdIn1,
                                            const Tensor &fwdOut) const {
  // create the new ops
  auto log  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);
  auto mul1 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul2 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Connect up the new ops
  log->connectInTensor(0, fwdIn0.id);
  log->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  log->outInfo(0) = log->inInfo(0);

  mul1->connectInTensor(0, fwdOut.id);
  mul1->connectInTensor(1, log->outTensor(0)->id);
  mul1->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul1->outInfo(0) = op->prettyNpOut(mul1->inInfo(0), mul1->inInfo(1));

  mul2->connectInTensor(0, gradIn.id);
  mul2->connectInTensor(1, mul1->outTensor(0)->id);
  mul2->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul2->outInfo(0) = op->prettyNpOut(mul2->inInfo(0), mul2->inInfo(1));

  return mul2->outTensor(0)->id;
}

namespace {
static PatternCreator<popart::PowArg1GradOpPattern>
    PowArg1GradOpPattern(PreAliasPatternType::PowArg1GradOp,
                         "PowArg1GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
