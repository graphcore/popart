// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE pipeline_unittest

#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/pipeline.hpp>

using namespace popart;

namespace {

class TestOp : public Op {
public:
  TestOp(bool preLoss, bool recompute, const Op::Settings &_settings)
      : Op(OperatorIdentifier("TestOps", "TestOp", 1), _settings) {
    scheduledPreLoss = preLoss ? ScheduledPreLoss::Yes : ScheduledPreLoss::No;
    settings.recomputeType =
        recompute ? RecomputeType::Recompute : RecomputeType::Checkpoint;
  }
  std::unique_ptr<Op> clone() const final {
    return std::make_unique<TestOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void setup() final { outInfo(0) = inInfo(0); }
};

class CopyOp : public Op {
public:
  CopyOp(const Op::Settings &settings)
      : Op(OperatorIdentifier("TestOps", "CopyOp", 1), settings) {}

  bool isIpuCopyOp() const final { return true; }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CopyOp>(*this);
  }
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void setup() final { outInfo(0) = inInfo(0); }
};

TensorId addStartTensor(Graph &g, Op::Settings settings) {
  TensorId t0 = g.getIr().createIntermediateTensorId("t0");
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  g.getTensors().addVarInit(t0, t0Info, static_cast<void *>(&t0Data));
  auto t0_c = g.getIr().createIntermediateTensorId(t0);
  g.createConnectedOp<CopyOp>({{0, t0}}, {{0, t0_c}}, settings);
  return t0_c;
}

} // namespace

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_0) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  auto op1 = g.createConnectedOp<TestOp>(
      {{0, t0}}, {{0, "t1"}}, true, false, settings);
  auto op2 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t2"}}, true, true, settings);
  auto op3 = g.createConnectedOp<TestOp>(
      {{0, "t2"}}, {{0, "t3"}}, true, false, settings);
  auto op4 = g.createConnectedOp<TestOp>(
      {{0, "t3"}}, {{0, "t4"}}, true, true, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "t4"}}, {{0, "t5"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);
  BOOST_REQUIRE(op1->settings.recomputeType == RecomputeType::Checkpoint);
  // op2 should be unchanged
  BOOST_REQUIRE(op2->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op3->settings.recomputeType == RecomputeType::Checkpoint);
  // op4 should be changed from Recompute to Checkpoint
  BOOST_REQUIRE(op4->settings.recomputeType == RecomputeType::Checkpoint);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_1) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto a0 = addStartTensor(g, settings);
  auto b0 = addStartTensor(g, settings);

  // PreLoss
  auto a_op1 = g.createConnectedOp<TestOp>(
      {{0, a0}}, {{0, "a1"}}, true, false, settings);
  auto a_op2 = g.createConnectedOp<TestOp>(
      {{0, "a1"}}, {{0, "a3"}}, true, true, settings);

  auto b_op1 = g.createConnectedOp<TestOp>(
      {{0, b0}}, {{0, "b1"}}, true, false, settings);
  auto b_op2 = g.createConnectedOp<TestOp>(
      {{0, "b1"}}, {{0, "b3"}}, true, true, settings);

  auto c_op = g.createConnectedOp<TestOp>(
      {{0, "a3"}, {1, "b3"}}, {{0, "c"}}, true, false, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "c"}}, {{0, "c1"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);

  BOOST_REQUIRE(a_op2->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(b_op2->settings.recomputeType == RecomputeType::Recompute);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_fail) {
  Ir ir;

  auto &g                 = ir.getMainGraph();
  Op::Settings settings0  = Op::Settings{g, "testOp"};
  settings0.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings0);
  g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, false, settings0);
  g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t2"}}, false, false, settings0);

  Op::Settings settings1  = Op::Settings{g, "testOp"};
  settings1.pipelineStage = 1;

  auto t3 = addStartTensor(g, settings1);
  g.createConnectedOp<TestOp>({{0, t3}}, {{0, "t4"}}, true, false, settings1);
  g.createConnectedOp<TestOp>(
      {{0, "t4"}}, {{0, "t5"}}, false, false, settings1);

  // More than one 'finalFwdStage' candidate
  BOOST_REQUIRE_THROW(Pipeline::setFinalFwdStageRecomputation(g),
                      internal_error);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_nofrontier) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  auto op1 =
      g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, true, settings);
  auto op2 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t2"}}, true, true, settings);
  // PostLoss
  g.createConnectedOp<TestOp>({{0, "t2"}}, {{0, "t3"}}, false, false, settings);

  // All pre-loss ops are marked recompute, so the frontier (from which
  // checkpoint annotations are forward-propagated) is empty.
  // Note: a call to Pipeline::setFinalFwdStageRecomputation is only expected
  // after a call to Pipeline::setRecompute, which should guarantee a
  // non-empty frontier.
  BOOST_REQUIRE_THROW(Pipeline::setFinalFwdStageRecomputation(g),
                      internal_error);
}

BOOST_AUTO_TEST_CASE(setfinalstagerecompute_nontopoorder) {
  Ir ir;

  auto &g                = ir.getMainGraph();
  Op::Settings settings  = Op::Settings{g, "testOp"};
  settings.pipelineStage = 0;

  auto t0 = addStartTensor(g, settings);

  // PreLoss
  auto op1 =
      g.createConnectedOp<TestOp>({{0, t0}}, {{0, "t1"}}, true, true, settings);
  auto op2 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "out"}}, true, true, settings);
  auto op3 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t3"}}, true, false, settings);
  op2->connectInTensor(1, "t3");
  auto op4 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t4"}}, true, true, settings);
  op3->connectInTensor(1, "t4");
  auto op5 = g.createConnectedOp<TestOp>(
      {{0, "t1"}}, {{0, "t5"}}, true, false, settings);
  op4->connectInTensor(1, "t5");
  // PostLoss
  g.createConnectedOp<TestOp>(
      {{0, "out"}}, {{0, "postLoss"}}, false, false, settings);

  Pipeline::setFinalFwdStageRecomputation(g);

  BOOST_REQUIRE(op1->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op2->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op4->settings.recomputeType == RecomputeType::Recompute);
  BOOST_REQUIRE(op3->settings.recomputeType == RecomputeType::Checkpoint);
  BOOST_REQUIRE(op5->settings.recomputeType == RecomputeType::Checkpoint);
}