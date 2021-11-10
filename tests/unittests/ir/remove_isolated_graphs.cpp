// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE pipeline_unittest

#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/pipeline.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(remove_all) {
  Ir ir;

  BOOST_REQUIRE(ir.getAllGraphs().size() == 1);

  auto &g = ir.getMainGraph();

  ir.createGraph(GraphId("foo"));
  ir.createGraph(GraphId("bar"));

  BOOST_REQUIRE(ir.getAllGraphs().size() == 3);

  ir.removeIsolatedGraphs();

  BOOST_REQUIRE(ir.getAllGraphs().size() == 1);
}

BOOST_AUTO_TEST_CASE(doesnt_remove_one) {
  Ir ir;

  BOOST_REQUIRE(ir.getAllGraphs().size() == 1);

  auto &g = ir.getMainGraph();

  ir.createGraph(GraphId("foo"));
  auto &sg = ir.createGraph(GraphId("bar"));

  BOOST_REQUIRE(ir.getAllGraphs().size() == 3);

  Op::Settings settings = Op::Settings{g, "testOp"};
  g.createConnectedOp<CallOp>(
      {}, {}, Onnx::CustomOperators::Call_1, sg, settings);

  ir.removeIsolatedGraphs();

  BOOST_REQUIRE(ir.getAllGraphs().size() == 2);
}