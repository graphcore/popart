// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_tensortestwrapper

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>

#include <iostream>
#include <sstream>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>

#define private public
#include <testutil/irquery/irquery.hpp>
#undef private

#include <testutil/irquery/mock_testfailuretriggerer.hpp>
#include <testutil/irquery/testop.hpp>

using namespace popart;
using namespace popart::irquery;

namespace {
auto _ = trompeloeil::_;
}

BOOST_AUTO_TEST_CASE(tensortestwrapper_index) {

  // Create this graph, where every op is an IdentityOp:
  //
  //         [t0]
  //          |  \
  //          |   |
  //          v   v
  //         op0 op1
  //          |   |
  //         [t1][t2]
  //          |
  //          v
  //         op2
  //          |
  //         [t3]

  Ir ir;
  auto &mainGraph        = ir.getMainGraph();
  auto &mainGraphTensors = mainGraph.getTensors();
  mainGraphTensors.addActGrad("t0");
  auto t0 = mainGraphTensors.get("t0");

  auto op0 = mainGraph.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), "t0"}},
      {{IdentityOp::getOutIndex(), "t1"}},
      Onnx::Operators::Identity_1,
      Op::Settings{mainGraph, "TestOp"});

  auto op1 = mainGraph.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), "t0"}},
      {{IdentityOp::getOutIndex(), "t2"}},
      Onnx::Operators::Identity_1,
      Op::Settings{mainGraph, "TestOp"});

  auto op2 = mainGraph.createConnectedOp<IdentityOp>(
      {{IdentityOp::getInIndex(), "t1"}},
      {{IdentityOp::getOutIndex(), "t3"}},
      Onnx::Operators::Identity_1,
      Op::Settings{mainGraph, "TestOp"});

  // Test t0's consumer ops are op0, op1
  TensorTestWrapper t0Tw{ir, t0};
  std::multiset<Op *> t0ConsumersExpected{op0, op1};
  auto t0Unwrap = t0Tw.consumers().unwrap();
  std::multiset<Op *> t0ConsumersActual(t0Unwrap.begin(), t0Unwrap.end());
  BOOST_REQUIRE_MESSAGE(
      t0ConsumersExpected == t0ConsumersActual,
      logging::format(
          "Expected consumers {}, actual {}",
          logging::join(
              t0ConsumersExpected.begin(), t0ConsumersExpected.end(), ", "),
          logging::join(
              t0ConsumersActual.begin(), t0ConsumersActual.end(), ", ")));

  // Test t1's consumers are op2 (get t1 via)
  auto t1 = mainGraphTensors.get("t1");
  TensorTestWrapper t1Tw{ir, t1};
  std::multiset<Op *> t1ConsumersExpected{op2};
  auto t1Unwrap = t1Tw.consumers().unwrap();
  std::multiset<Op *> t1ConsumersActual(t1Unwrap.begin(), t1Unwrap.end());
  BOOST_REQUIRE(t1ConsumersExpected == t1ConsumersActual);
}
