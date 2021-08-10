// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE recompute_unittest
#include <boost/test/unit_test.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/recompute.hpp>

using namespace popart;

namespace {
class TestOp : public Op {
public:
  TestOp(bool recompute_checkpoint, const Op::Settings &_settings)
      : Op(OperatorIdentifier("TestOps", "TestOp", 1), _settings) {
    scheduledPreLoss       = ScheduledPreLoss::Yes;
    toLoss                 = PathToLoss::Yes;
    settings.recomputeType = recompute_checkpoint ? RecomputeType::Checkpoint
                                                  : RecomputeType::Undefined;
  }
  std::unique_ptr<Op> clone() const final {
    return std::make_unique<TestOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void setup() final { outInfo(0) = inInfo(0); }
};
} // namespace

BOOST_AUTO_TEST_CASE(recomputealltest) {
  // Test the following graph:
  //         /->op2(checkpoint)\
    // a0->op1                     ->op4->loss
  //         \->op3(checkpint) /
  Ir ir;
  auto &g               = ir.getMainGraph();
  Op::Settings settings = Op::Settings{g, "testOp"};
  // Create input
  auto a0 = ir.createIntermediateTensorId("a0");
  TensorInfo a0Info{DataType::INT32, {}};
  int32_t a0Data[] = {5};
  g.getTensors().addVarInit(a0, a0Info, static_cast<void *>(&a0Data));

  auto op1 =
      g.createConnectedOp<TestOp>({{0, a0}}, {{0, "op1"}}, false, settings);
  auto op2 =
      g.createConnectedOp<TestOp>({{0, "op1"}}, {{0, "op2"}}, true, settings);
  auto op3 =
      g.createConnectedOp<TestOp>({{0, "op1"}}, {{0, "op3"}}, true, settings);
  auto op4 = g.createConnectedOp<TestOp>(
      {{0, "op2"}, {1, "op3"}}, {{0, "op4"}}, false, settings);
  auto loss =
      g.createConnectedOp<TestOp>({{0, "op4"}}, {{0, "loss"}}, true, settings);
  ir.setFinalLoss(loss->outId(0));

  recompute::annotateRecomputeAll(g);
  BOOST_REQUIRE(op1->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op2->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op3->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op4->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(loss->settings.recomputeType == RecomputeType::Checkpoint);
}