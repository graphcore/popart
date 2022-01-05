// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_graphtestwrapper

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>

#include <iostream>
#include <sstream>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/util.hpp>

#define private public
#include <testutil/irquery/irquery.hpp>
#undef private

#include <testutil/irquery/mock_testfailuretriggerer.hpp>
#include <testutil/irquery/testop.hpp>

using namespace popart;
using namespace popart::irquery;

namespace {

auto _ = trompeloeil::_;

/**
 * Add this subgraph to an IR:
 *
 *       [A/i0]                  [A/i1][A/i2]
 *        |                       |     |
 * .------|------[subgraph "A"]---|-----|-----.
 * |      |                       |     |     |
 * |   #1 v                       |     |     |
 * |   +---------------+          |     |     |
 * |   | testOp0       |          |     |     |
 * |   +---------------+          |     |     |
 * |   #1 |                       |     |     |
 * |      |                       |     |     |
 * |    [A/tmp0] .----------------'     |     |
 * |      |      |                      |     |
 * |   #1 v   #2 v                      |     |
 * |   +---------------+                |     |
 * |   | testOp1       |                |     |
 * |   +---------------+                |     |
 * |   #1 | #2 |                        |     |
 * |      |    |                        |     |
 * |[A/tmp1][A/tmp2]                    |     |
 * |      |    |    .-------------------'     |
 * |      |    |    |                         |
 * |   #1 v #2 v #3 v                         |
 * |   +---------------+                      |
 * |   | testOp2       |                      |
 * |   +---------------+                      |
 * |   #1 | #2 | #3 |                         |
 * |      |    |    |                         |
 * '------|----|----|-------------------------'
 *        v    v    v
 *     [A/o0][A/o1][A/o2]
 *
 * Returns: ops, inputs & outputs.
 **/
std::tuple<std::vector<Op *>, std::vector<TensorId>, std::vector<TensorId>>
addTestGraph(Ir &ir) {

  // Tensor info for tensors in the IR.
  TensorInfo tInfo{DataType::INT32, {}};
  int32_t tData[] = {5};

  // Create the subgraph.
  auto &subgraphA       = ir.createGraph(GraphId("A"));
  Op::Settings settings = Op::Settings{subgraphA, "TestOp"};

  // Create subgraph A.
  subgraphA.addInput(addScope(subgraphA, "i0"), tInfo);
  subgraphA.addInput(addScope(subgraphA, "i1"), tInfo);
  subgraphA.addInput(addScope(subgraphA, "i2"), tInfo);

  // Add SimpleTestOps.
  auto testOp0 = subgraphA.createOp<TestOp>(settings);
  auto testOp1 = subgraphA.createOp<TestOp>(settings);
  auto testOp2 = subgraphA.createOp<TestOp>(settings);

  testOp0->connectInTensor(1, addScope(subgraphA, "i0"));
  testOp0->createAndConnectOutTensor(1, addScope(subgraphA, "tmp0"));
  testOp0->setup();

  testOp1->connectInTensor(1, addScope(subgraphA, "tmp0"));
  testOp1->connectInTensor(2, addScope(subgraphA, "i1"));
  testOp1->createAndConnectOutTensor(1, addScope(subgraphA, "tmp1"));
  testOp1->createAndConnectOutTensor(2, addScope(subgraphA, "tmp2"));
  testOp1->setup();

  testOp2->connectInTensor(1, addScope(subgraphA, "tmp1"));
  testOp2->connectInTensor(2, addScope(subgraphA, "tmp2"));
  testOp2->connectInTensor(3, addScope(subgraphA, "i2"));
  testOp2->createAndConnectOutTensor(1, addScope(subgraphA, "o0"));
  testOp2->createAndConnectOutTensor(2, addScope(subgraphA, "o1"));
  testOp2->createAndConnectOutTensor(3, addScope(subgraphA, "o2"));
  testOp2->setup();

  subgraphA.markAsOutput(addScope(subgraphA, "o0"));
  subgraphA.markAsOutput(addScope(subgraphA, "o1"));
  subgraphA.markAsOutput(addScope(subgraphA, "o2"));

  return {{testOp0, testOp1, testOp2},
          {addScope(subgraphA, "i0"),
           addScope(subgraphA, "i1"),
           addScope(subgraphA, "i2")},
          {addScope(subgraphA, "o0"),
           addScope(subgraphA, "o1"),
           addScope(subgraphA, "o2")}};
}
} // namespace

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_ops) {

  Ir ir;
  auto ops = std::get<0>(addTestGraph(ir));
  GraphTestWrapper tw{ir, GraphId("A")};

  // Check ops() is populated correctly.
  BOOST_REQUIRE("subgraph 'A'" == tw.ops().srcObjDescr);
  BOOST_REQUIRE(ops == tw.ops().unwrap());
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_inputs) {

  Ir ir;
  auto inputs = std::get<1>(addTestGraph(ir));
  GraphTestWrapper tw{ir, GraphId("A")};

  // Check inputs() is populated correctly.
  BOOST_REQUIRE("subgraph 'A'" == tw.inputs().srcObjDescr);
  BOOST_REQUIRE("input" == tw.inputs().mapTypeDescrSingular);
  BOOST_REQUIRE("inputs" == tw.inputs().mapTypeDescrPlural);
  BOOST_REQUIRE(inputs.size() == tw.inputs().unwrap().size());
  for (size_t s = 0; s < inputs.size(); ++s) {
    BOOST_REQUIRE(ir.getGraph(GraphId("A")).getTensors().get(inputs.at(s)) ==
                  tw.inputs().unwrap().at(s));
  }
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_outputs) {

  Ir ir;
  auto outputs = std::get<2>(addTestGraph(ir));
  GraphTestWrapper tw{ir, GraphId("A")};

  // Check outputs() is populated correctly.
  BOOST_REQUIRE("subgraph 'A'" == tw.outputs().srcObjDescr);
  BOOST_REQUIRE("output" == tw.outputs().mapTypeDescrSingular);
  BOOST_REQUIRE("outputs" == tw.outputs().mapTypeDescrPlural);
  BOOST_REQUIRE(outputs.size() == tw.outputs().unwrap().size());
  for (size_t s = 0; s < outputs.size(); ++s) {
    BOOST_REQUIRE(ir.getGraph(GraphId("A")).getTensors().get(outputs.at(s)) ==
                  tw.outputs().unwrap().at(s));
  }
}
