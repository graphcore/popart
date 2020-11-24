// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/atanh.hpp>
#include <popart/op/div.hpp>
#include <popart/op/log.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/subtract.hpp>
#include <popart/patterns/atanhoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool AtanhOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<AtanhOp>();
}

std::vector<const Tensor *> AtanhOpPattern::touches(Op *) const { return {}; }

// atanh(x) = 1/2 ln( (1 + x) / (1 - x) ). Defined for (-1, +1)

bool AtanhOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(AtanhOp::getInIndex());
  auto output = op->outTensor(AtanhOp::getOutIndex());

  TensorInfo onesInfo(input->info.dataType(), {1});
  std::vector<float> onesData(1, 1.0f);
  auto onesId = op->getIr().createIntermediateTensorId("ones");
  op->getGraph().getTensors().addConstInit(
      onesId, onesInfo, reinterpret_cast<void *>(onesData.data()));

  TensorInfo halfInfo(input->info.dataType(), {1});
  std::vector<float> halfData(1, 0.5f);
  auto halfId = op->getIr().createIntermediateTensorId("half");
  op->getGraph().getTensors().addConstInit(
      halfId, halfInfo, reinterpret_cast<void *>(halfData.data()));

  // create the new ops
  auto sub = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sub, op);
  auto add = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  auto div = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Div, op);
  auto log = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);
  auto mul = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Mul, op);

  // Remove the AtanhOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  sub->connectInTensor(SubtractOp::getArg0InIndex(), onesId);
  sub->connectInTensor(SubtractOp::getArg1InIndex(), input->id);
  sub->createAndConnectOutTensor(
      SubtractOp::getOutIndex(),
      input->getIr().createIntermediateTensorId("sub"));
  sub->setup();

  add->connectInTensor(AddOp::getArg0InIndex(), onesId);
  add->connectInTensor(AddOp::getArg1InIndex(), input->id);
  add->createAndConnectOutTensor(
      AddOp::getOutIndex(), input->getIr().createIntermediateTensorId("add"));
  add->setup();

  div->connectInTensor(DivOp::getArg0InIndex(),
                       add->outTensor(AddOp::getOutIndex())->id);
  div->connectInTensor(DivOp::getArg1InIndex(),
                       sub->outTensor(SubtractOp::getOutIndex())->id);
  div->createAndConnectOutTensor(
      DivOp::getOutIndex(), input->getIr().createIntermediateTensorId("div"));
  div->setup();

  log->connectInTensor(LogOp::getInIndex(),
                       div->outTensor(DivOp::getOutIndex())->id);
  log->createAndConnectOutTensor(
      LogOp::getOutIndex(), input->getIr().createIntermediateTensorId("log"));
  log->setup();

  mul->connectInTensor(MulOp::getArg0InIndex(), halfId);
  mul->connectInTensor(MulOp::getArg1InIndex(),
                       log->outTensor(LogOp::getOutIndex())->id);
  mul->connectOutTensor(MulOp::getOutIndex(), output->id);
  mul->setup();

  return true;
}

namespace {
static PatternCreator<popart::AtanhOpPattern>
    AtanhOpPattern(PreAliasPatternType::AtanhOpPattern, "AtanhOpPattern");

} // namespace

} // namespace popart
