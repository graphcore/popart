// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/expm1.hpp>
#include <popart/op/mul.hpp>
#include <popart/patterns/expm1gradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/op.hpp"
#include "popart/opdebuginfo.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensors.hpp"

namespace popart {

bool Expm1GradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<Expm1GradOp>();
}

std::vector<const Tensor *> Expm1GradOpPattern::touches(Op *) const {
  return {};
}

// fwd_out is exp(x) - 1, where x is input tensor.
// But we want grad_out to be grad_in * exp(x).
// Therefore we add 1 to fwd_out:
// grad_out = grad_in * ( fwd_out + 1 )
bool Expm1GradOpPattern::apply(Op *op) const {
  auto grad_in  = op->inTensor(Expm1GradOp::getGradInIndex());
  auto fwd_out  = op->inTensor(Expm1GradOp::getFwdOutInIndex());
  auto grad_out = op->outTensor(Expm1GradOp::getOutIndex());

  TensorInfo onesInfo(fwd_out->info.dataType(), {1});
  std::vector<float> onesData(1, 1.0f);
  auto onesId = op->getIr().createIntermediateTensorId("ones");
  op->getGraph().getTensors().addConstInit(
      onesId,
      onesInfo,
      reinterpret_cast<void *>(onesData.data()),
      op->getDebugInfo());

  // create the new ops
  auto mul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);
  auto add = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);

  // Remove the Expm1GradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  add->connectInTensor(0, fwd_out->id);
  add->connectInTensor(1, onesId);
  add->createAndConnectOutTensor(
      0, fwd_out->getIr().createIntermediateTensorId(fwd_out->id));
  add->setup();

  mul->connectInTensor(MulOp::getArg0InIndex(), grad_in->id);
  mul->connectInTensor(MulOp::getArg1InIndex(), add->outTensor(0)->id);
  mul->connectOutTensor(MulOp::getOutIndex(), grad_out->id);
  mul->setup();

  return true;
}

namespace {
static PatternCreator<Expm1GradOpPattern>
    Expm1GradOpPattern("Expm1GradOp", true, true);
}

} // namespace popart
