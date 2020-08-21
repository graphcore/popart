// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/acosh.hpp>
#include <popart/op/add.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/patterns/acoshoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool AcoshOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<AcoshOp>();
}

std::vector<const Tensor *> AcoshOpPattern::touches(Op *) const { return {}; }

// acosh(x) = ln(x + sqrt(x^2 - 1) ); Defined for [1, +inf)

bool AcoshOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(AcoshOp::getInIndex());
  auto output = op->outTensor(AcoshOp::getOutIndex());

  TensorInfo onesInfo(input->info.dataType(), {1});
  std::vector<float> onesData(1, 1.0f);
  auto onesId = op->getIr().createIntermediateTensorId("ones");
  op->getGraph().getTensors().addConstInit(
      onesId, onesInfo, reinterpret_cast<void *>(onesData.data()));

  TensorInfo twosInfo(input->info.dataType(), {1});
  std::vector<float> twosData(1, 2.0f);
  auto twosId = op->getIr().createIntermediateTensorId("twos");
  op->getGraph().getTensors().addConstInit(
      twosId, twosInfo, reinterpret_cast<void *>(twosData.data()));

  // create the new ops
  auto pow  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Pow, op);
  auto sub  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sub, op);
  auto sqrt = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sqrt, op);
  auto add  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  auto log  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);

  // Remove the AcoshOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  pow->connectInTensor(0, input->id);
  pow->connectInTensor(1, twosId);
  pow->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId(input->id));
  pow->setup();

  sub->connectInTensor(0, pow->outTensor(0)->id);
  sub->connectInTensor(1, onesId);
  sub->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId(input->id));
  sub->setup();

  sqrt->connectInTensor(0, sub->outTensor(0)->id);
  sqrt->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId(input->id));
  sqrt->setup();

  add->connectInTensor(AddOp::getArg0InIndex(), input->id);
  add->connectInTensor(AddOp::getArg1InIndex(),
                       sqrt->outTensor(SqrtOp::getOutIndex())->id);
  add->createAndConnectOutTensor(
      SqrtOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  add->setup();

  log->connectInTensor(0, add->outTensor(0)->id);
  log->connectOutTensor(0, output->id);
  log->setup();

  return true;
}

namespace {
static PatternCreator<popart::AcoshOpPattern>
    AcoshOpPattern(PreAliasPatternType::AcoshOpPattern, "AcoshOpPattern");

} // namespace

} // namespace popart
