// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/acos.hpp>
#include <popart/patterns/acosoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool AcosOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<AcosOp>();
}

std::vector<const Tensor *> AcosOpPattern::touches(Op *) const { return {}; }

// acos(x) = pi / 2 - asin(x)
bool AcosOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(AcosOp::getInIndex());
  auto output = op->outTensor(AcosOp::getOutIndex());

  TensorInfo piover2Info(input->info.dataType(), {1});
  std::vector<float> piover2Data(1, M_PI_2);
  auto piover2Id = op->getIr().createIntermediateTensorId("piover2");
  op->getGraph().getTensors().addConstInit(
      piover2Id, piover2Info, reinterpret_cast<void *>(piover2Data.data()));

  // create the new ops
  auto asin = makeReplacementOpInIr(Onnx::Operators::Asin_7, op);
  auto sub  = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Sub, op);

  // Remove the AcosOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->getGraph().eraseOp(op->id);

  // Connect up the new ops
  asin->connectInTensor(0, input->id);
  asin->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId(input->id));
  asin->setup();

  sub->connectInTensor(0, piover2Id);
  sub->connectInTensor(1, asin->outTensor(0)->id);
  sub->connectOutTensor(0, output->id);
  sub->setup();

  return true;
}

namespace {
static PatternCreator<popart::AcosOpPattern>
    AcosOpPattern(PreAliasPatternType::AcosOpPattern, "AcosOpPattern");

} // namespace

} // namespace popart
