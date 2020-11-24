// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/asinh.hpp>
#include <popart/op/log.hpp>
#include <popart/op/pow.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/patterns/asinhoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool AsinhOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<AsinhOp>();
}

std::vector<const Tensor *> AsinhOpPattern::touches(Op *) const { return {}; }

// asinh(x) = ln(x + sqrt(x^2 + 1))

bool AsinhOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(AsinhOp::getInIndex());
  auto output = op->outTensor(AsinhOp::getOutIndex());

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
  auto add  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  auto add2 = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Add, op);
  auto sqrt = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sqrt, op);
  auto log  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);

  // Remove the AsinhOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  pow->connectInTensor(PowOp::getArg0InIndex(), input->id);
  pow->connectInTensor(PowOp::getArg1InIndex(), twosId);
  pow->createAndConnectOutTensor(
      PowOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  pow->setup();

  add2->connectInTensor(AddOp::getArg0InIndex(),
                        pow->outTensor(PowOp::getOutIndex())->id);
  add2->connectInTensor(AddOp::getArg1InIndex(), onesId);
  add2->createAndConnectOutTensor(
      AddOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  add2->setup();

  sqrt->connectInTensor(SqrtOp::getInIndex(),
                        add2->outTensor(AddOp::getOutIndex())->id);
  sqrt->createAndConnectOutTensor(
      SqrtOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  sqrt->setup();

  add->connectInTensor(AddOp::getArg0InIndex(), input->id);
  add->connectInTensor(AddOp::getArg1InIndex(),
                       sqrt->outTensor(SqrtOp::getOutIndex())->id);
  add->createAndConnectOutTensor(
      AddOp::getOutIndex(),
      input->getIr().createIntermediateTensorId(input->id));
  add->setup();

  log->connectInTensor(LogOp::getInIndex(),
                       add->outTensor(AddOp::getOutIndex())->id);
  log->connectOutTensor(LogOp::getOutIndex(), output->id);
  log->setup();

  return true;
}

namespace {
static PatternCreator<popart::AsinhOpPattern>
    AsinhOpPattern(PreAliasPatternType::AsinhOpPattern, "AsinhOpPattern");

} // namespace

} // namespace popart
