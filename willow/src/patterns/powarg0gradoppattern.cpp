// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/pow.hpp>
#include <popart/op/subtract.hpp>
#include <popart/optimizer.hpp>
#include <popart/patterns/powarg0gradoppattern.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool PowArg0GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<PowArg0GradOp>();
}

// grad_out = grad_in *arg1 * arg0 ^(arg1 - 1)
TensorId PowArg0GradOpPattern::makeAllReplacementOps(Op *op,
                                                     Tensor *grad_in,
                                                     Tensor *fwd_in0,
                                                     Tensor *fwd_in1,
                                                     Tensor *fwd_out) const {
  // Create a 1-dim constant tensor of value 1.0f with same type as fwd_in1
  TensorInfo resultInfo(fwd_in1->info.dataType(), {1});
  std::vector<float> resultData(1, 1.0f);
  auto onesId = op->getIr().createIntermediateTensorId("ones");
  op->getGraph().getTensors().addConstInit(
      onesId, resultInfo, reinterpret_cast<void *>(resultData.data()));

  // create the new ops
  auto sub   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sub, op);
  auto pow   = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Pow, op);
  auto mul_1 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto mul_2 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Connect up the new ops
  sub->connectInTensor(0, fwd_in1->id);
  sub->connectInTensor(1, onesId);
  sub->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  sub->outInfo(0) = op->prettyNpOut(sub->inInfo(0), sub->inInfo(1));

  pow->connectInTensor(0, fwd_in0->id);
  pow->connectInTensor(1, sub->outTensor(0)->id);
  pow->createAndConnectOutTensor(
      0, grad_in->getIr().createIntermediateTensorId(grad_in->id));
  pow->outInfo(0) = op->prettyNpOut(pow->inInfo(0), pow->inInfo(1));

  mul_1->connectInTensor(0, fwd_in1->id);
  mul_1->connectInTensor(1, pow->outTensor(0)->id);
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
static PatternCreator<PowArg0GradOpPattern>
    PowArg0GradOpPattern(PreAliasPatternType::PowArg0GradOp,
                         "PowArg0GradOp",
                         /* enabled = */ true,
                         /* mandatory = */ true);
}

} // namespace popart
