// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/div.hpp>
#include <popart/op/log1p.hpp>
#include <popart/patterns/log1pgradoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensors.hpp"

namespace popart {

bool Log1pGradOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<Log1pGradOp>();
}

std::vector<const Tensor *> Log1pGradOpPattern::touches(Op *) const {
  return {};
}

// d/dx log(x+1) = 1/(x+1)
// grad_out = grad_in / (fwd_in + 1)
bool Log1pGradOpPattern::apply(Op *op) const {

  auto grad_in  = op->inTensor(Log1pGradOp::getGradInIndex());
  auto fwd_in   = op->inTensor(Log1pGradOp::getFwdArgInIndex());
  auto grad_out = op->outTensor(Log1pGradOp::getOutIndex());

  TensorInfo onesInfo(fwd_in->info.dataType(), {1});
  std::vector<float> onesData(1, 1.0f);
  auto onesId = op->getIr().createIntermediateTensorId("ones");
  op->getGraph().getTensors().addConstInit(
      onesId, onesInfo, reinterpret_cast<void *>(onesData.data()));

  // create the new ops
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);
  auto add = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);

  // Remove the Log1pGradOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  add->connectInTensor(0, fwd_in->id);
  add->connectInTensor(1, onesId);
  add->createAndConnectOutTensor(
      0, fwd_in->getIr().createIntermediateTensorId(fwd_in->id));
  add->setup();

  // Connect up the new ops
  div->connectInTensor(DivOp::getArg0InIndex(), grad_in->id);
  div->connectInTensor(DivOp::getArg1InIndex(), add->outTensor(0)->id);
  div->connectOutTensor(DivOp::getOutIndex(), grad_out->id);
  div->setup();

  return true;
}

namespace {
static PatternCreator<Log1pGradOpPattern>
    Log1pGradOpPattern("Log1pGradOp", true, true);
}

} // namespace popart
