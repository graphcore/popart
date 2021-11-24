// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE overlapio_unittest

#include <testutil/irquery/irquery.hpp>
#include <testutil/test_graphs/graph_test_models.hpp>

#include <boost/test/unit_test.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/op/exchange/hostcopy.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/tensor.hpp>
#include <popart/transforms/overlapio.hpp>

using namespace popart;
using namespace popart::irquery;

namespace {

template <typename T>
bool checkSchedule(const std::vector<Op *> &schedule, size_t &j) {
  return schedule.at(j++)->isConvertibleTo<T>();
}

// Test overlap IO graph when all inputs and outputs are overlapped for the
// inner loop
BOOST_AUTO_TEST_CASE(OverlapInnerLoop) {
  GraphTestModel3 model(ExchangeStrategy::OverlapInnerLoop,
                        ExchangeStrategy::OverlapInnerLoop,
                        ExchangeStrategy::OverlapInnerLoop);

  auto &ir = model.getIr();
  ir.applyTransform(OverlapIO::id(), ir.getMainGraph());
  ir.updateVertices();
  ir.setIsPrepared();

  ir.dotCheckpoint(DotCheck::Final);

  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
  auto tw_outerLoopOp = tw_mainGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getStepGraphName());
      },
      Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_mainGraph->unwrap().get().getOps().size(), 1);

  auto outerLoopOp    = tw_outerLoopOp->unwrap();
  GraphId stepGraphId = outerLoopOp->getCalledGraph().id;
  auto tw_stepGraph   = tw_ir.hasGraph(stepGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getInputIds().size(), 2);
  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getOutputIds().size(), 1);

  auto tw_innerLoopOp = tw_stepGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getAccumulationGraphName());
      },
      Require::MustBeTrue);

  auto innerLoopOp     = tw_innerLoopOp->unwrap();
  GraphId accumGraphId = innerLoopOp->getCalledGraph().id;
  auto tw_accumGraph   = tw_ir.hasGraph(accumGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getInputIds().size(), 5);
  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getOutputIds().size(), 4);

  auto stepGraphSchedule = tw_stepGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < stepGraphSchedule.size(); ++i) {
    logging::trace(
        "Step graph: {}: {}", i, stepGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<LoopOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
  }

  auto accumGraphSchedule = tw_accumGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < accumGraphSchedule.size(); ++i) {
    logging::trace(
        "Accum graph: {}: {}", i, accumGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
  }
}

// Test overlap IO graph when only input A is overlapped for the inner loop
// (see input A in GraphTestModel3)
BOOST_AUTO_TEST_CASE(OverlapInnerLoopA) {
  GraphTestModel3 model(ExchangeStrategy::OverlapInnerLoop,
                        ExchangeStrategy::JustInTime,
                        ExchangeStrategy::JustInTime);

  auto &ir = model.getIr();
  ir.applyTransform(OverlapIO::id(), ir.getMainGraph());
  ir.updateVertices();
  ir.setIsPrepared();

  ir.dotCheckpoint(DotCheck::Final);

  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
  auto tw_outerLoopOp = tw_mainGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getStepGraphName());
      },
      Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_mainGraph->unwrap().get().getOps().size(), 1);

  auto outerLoopOp    = tw_outerLoopOp->unwrap();
  GraphId stepGraphId = outerLoopOp->getCalledGraph().id;
  auto tw_stepGraph   = tw_ir.hasGraph(stepGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getInputIds().size(), 2);
  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getOutputIds().size(), 1);

  auto tw_innerLoopOp = tw_stepGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getAccumulationGraphName());
      },
      Require::MustBeTrue);

  auto innerLoopOp     = tw_innerLoopOp->unwrap();
  GraphId accumGraphId = innerLoopOp->getCalledGraph().id;
  auto tw_accumGraph   = tw_ir.hasGraph(accumGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getInputIds().size(), 3);
  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getOutputIds().size(), 2);

  auto stepGraphSchedule = tw_stepGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < stepGraphSchedule.size(); ++i) {
    logging::trace(
        "Step graph: {}: {}", i, stepGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<LoopOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
  }

  auto accumGraphSchedule = tw_accumGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < accumGraphSchedule.size(); ++i) {
    logging::trace(
        "Accum graph: {}: {}", i, accumGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
  }
}

// Test overlap IO graph when only output C is overlapped for the inner loop
// (see output C in GraphTestModel3)
BOOST_AUTO_TEST_CASE(OverlapInnerLoopC) {
  GraphTestModel3 model(ExchangeStrategy::JustInTime,
                        ExchangeStrategy::JustInTime,
                        ExchangeStrategy::OverlapInnerLoop);

  auto &ir = model.getIr();
  ir.applyTransform(OverlapIO::id(), ir.getMainGraph());
  ir.updateVertices();
  ir.setIsPrepared();

  ir.dotCheckpoint(DotCheck::Final);

  IrTestWrapper tw_ir{ir};
  auto tw_mainGraph = tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
  auto tw_outerLoopOp = tw_mainGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getStepGraphName());
      },
      Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_mainGraph->unwrap().get().getOps().size(), 1);

  auto outerLoopOp    = tw_outerLoopOp->unwrap();
  GraphId stepGraphId = outerLoopOp->getCalledGraph().id;
  auto tw_stepGraph   = tw_ir.hasGraph(stepGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getInputIds().size(), 2);
  BOOST_REQUIRE_EQUAL(tw_stepGraph->unwrap().get().getOutputIds().size(), 1);

  auto tw_innerLoopOp = tw_stepGraph->ops().hasOp<LoopOp>(
      [&](auto &tw_op) -> bool {
        return tw_op.unwrap()->getCalledGraphs().at(0)->id ==
               GraphId(MainLoops::getAccumulationGraphName());
      },
      Require::MustBeTrue);

  auto innerLoopOp     = tw_innerLoopOp->unwrap();
  GraphId accumGraphId = innerLoopOp->getCalledGraph().id;
  auto tw_accumGraph   = tw_ir.hasGraph(accumGraphId, Require::MustBeTrue);

  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getInputIds().size(), 3);
  BOOST_REQUIRE_EQUAL(tw_accumGraph->unwrap().get().getOutputIds().size(), 2);

  auto stepGraphSchedule = tw_stepGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < stepGraphSchedule.size(); ++i) {
    logging::trace(
        "Step graph: {}: {}", i, stepGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<LoopOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(stepGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(stepGraphSchedule, j));
  }

  auto accumGraphSchedule = tw_accumGraph->unwrap().get().getOpSchedule(
      {}, RequireOptimalSchedule::Yes);
  for (size_t i = 0; i < accumGraphSchedule.size(); ++i) {
    logging::trace(
        "Accum graph: {}: {}", i, accumGraphSchedule.at(i)->debugName());
  }

  {
    size_t j = 0;
    BOOST_REQUIRE(checkSchedule<HostStoreOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<InitOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<HostLoadOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<AddOp>(accumGraphSchedule, j));
    BOOST_REQUIRE(checkSchedule<IoTileCopyOp>(accumGraphSchedule, j));
  }
}

} // namespace
