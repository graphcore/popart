// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/pow.hpp>
#include <popart/op/subtract.hpp>
#include <popart/patterns/powarg0gradoppattern.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool PowArg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<PowArg0GradOp>();
}

// grad_out = grad_in *arg1 * arg0 ^(arg1 - 1)
TensorId
PowArg0GradOpPattern::makeAllReplacementOps(Op *op,
                                            Ir *ir,
                                            const Tensor &gradIn,
                                            const Tensor &fwdIn0,
                                            const Tensor &fwdIn1,
                                            const Tensor &fwdOut) const {
  // Create a 1-dim constant tensor of value 1.0f with same type as fwd_in1
  TensorInfo resultInfo(fwdIn1.info.dataType(), {1});
  auto onesId = ir->createIntermediateTensorId("ones");

  addConstInitFromFloat(1.0f, onesId, resultInfo, op->getGraph().getTensors());

  // create the new ops
  auto sub  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sub, op);
  auto pow  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Pow, op);
  auto mul1 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul2 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Connect up the new ops
  sub->connectInTensor(0, fwdIn1.id);
  sub->connectInTensor(1, onesId);
  sub->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  sub->outInfo(0) = op->prettyNpOut(sub->inInfo(0), sub->inInfo(1));

  pow->connectInTensor(0, fwdIn0.id);
  pow->connectInTensor(1, sub->outTensor(0)->id);
  pow->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  pow->outInfo(0) = op->prettyNpOut(pow->inInfo(0), pow->inInfo(1));

  mul1->connectInTensor(0, fwdIn1.id);
  mul1->connectInTensor(1, pow->outTensor(0)->id);
  mul1->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul1->outInfo(0) = op->prettyNpOut(mul1->inInfo(0), mul1->inInfo(1));

  mul2->connectInTensor(0, gradIn.id);
  mul2->connectInTensor(1, mul1->outTensor(0)->id);
  mul2->createAndConnectOutTensor(0, ir->createIntermediateTensorId(gradIn.id));
  mul2->outInfo(0) = op->prettyNpOut(mul2->inInfo(0), mul2->inInfo(1));

  return mul2->outTensor(0)->id;
}

namespace {
static PatternCreator<PowArg0GradOpPattern>
    PowArg0GradOpPattern("PowArg0GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
