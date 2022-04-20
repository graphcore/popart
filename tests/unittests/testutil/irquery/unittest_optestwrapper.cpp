// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_optestwrapper

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include "popart/datatype.hpp"
#include "popart/graphid.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#include <testutil/irquery/irquery.hpp> // IWYU pragma: keep
#undef private

#include <testutil/irquery/testop.hpp>

using namespace popart;
using namespace popart::irquery;

BOOST_AUTO_TEST_CASE(optestwrapper_inputs_outputs) {

  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};

  // Create this graph:
  // .---------------[main graph]---------------.
  // |     [i0]              [i1]  [i2](=inputs)|
  // |      |                 |     |           |
  // |   #1 v                 |     |           |
  // |   +---------------+    |     |           |
  // |   | testOp0       |    |     |           |
  // |   +---------------+    |     |           |
  // |   #1 |                 |     |           |
  // |      |                 |     |           |
  // |    [tmp0] .------------'     |           |
  // |      |    |                  |           |
  // |   #1 v #2 v                  |           |
  // |   +---------------+          |           |
  // |   | testOp1       |          |           |
  // |   +---------------+          |           |
  // |   #1 | #2 |                  |           |
  // |      |    |                  |           |
  // |   [tmp1][tmp2] .-------------'           |
  // |      |    |    |                         |
  // |      |    |    |                         |
  // |   #1 v #2 v #3 v                         |
  // |   +---------------+                      |
  // |   | testOp2       |                      |
  // |   +---------------+                      |
  // |   #1 | #2 | #3 |                         |
  // |      |    |    |                         |
  // |      v    v    v                         |
  // |     [o0] [o1] [o2]             (=outputs)|
  // '------------------------------------------'

  // Add inputs.
  mainGraph.getTensors().addVarInit("i0", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("i1", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("i2", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOps.
  auto testOp0 = mainGraph.createOp<TestOp>(settings);
  auto testOp1 = mainGraph.createOp<TestOp>(settings);
  auto testOp2 = mainGraph.createOp<TestOp>(settings);

  testOp0->connectInTensor(1, "i0");
  testOp0->createAndConnectOutTensor(1, "tmp0");
  testOp0->setup();

  testOp1->connectInTensor(1, "tmp0");
  testOp1->connectInTensor(2, "i1");
  testOp1->createAndConnectOutTensor(1, "tmp1");
  testOp1->createAndConnectOutTensor(2, "tmp2");
  testOp1->setup();

  testOp2->connectInTensor(1, "tmp1");
  testOp2->connectInTensor(2, "tmp2");
  testOp2->connectInTensor(3, "i2");
  testOp2->createAndConnectOutTensor(1, "o0");
  testOp2->createAndConnectOutTensor(2, "o1");
  testOp2->createAndConnectOutTensor(3, "o2");
  testOp2->setup();

  // Check inputs() is populated correctly for testOp1.
  OpTestWrapper<Op> testOp1Tw{ir, testOp1};
  std::cout << testOp1Tw.inputs().srcObjDescr << std::endl;
  BOOST_REQUIRE("101 (TestOps.TestOp:1)" == testOp1Tw.inputs().srcObjDescr);
  BOOST_REQUIRE("input" == testOp1Tw.inputs().mapTypeDescrSingular);
  BOOST_REQUIRE("inputs" == testOp1Tw.inputs().mapTypeDescrPlural);
  BOOST_REQUIRE(2 == testOp1Tw.inputs().unwrap().size());
  BOOST_REQUIRE(mainGraph.getTensors().get("tmp0") ==
                testOp1Tw.inputs().unwrap().at(1));
  BOOST_REQUIRE(mainGraph.getTensors().get("i1") ==
                testOp1Tw.inputs().unwrap().at(2));

  // Check outputs() is populated correctly for testOp2.
  OpTestWrapper<Op> testOp2Tw{ir, testOp2};
  BOOST_REQUIRE("102 (TestOps.TestOp:1)" == testOp2Tw.outputs().srcObjDescr);
  BOOST_REQUIRE("output" == testOp2Tw.outputs().mapTypeDescrSingular);
  BOOST_REQUIRE("outputs" == testOp2Tw.outputs().mapTypeDescrPlural);
  BOOST_REQUIRE(3 == testOp2Tw.outputs().unwrap().size());
  BOOST_REQUIRE(mainGraph.getTensors().get("o0") ==
                testOp2Tw.outputs().unwrap().at(1));
  BOOST_REQUIRE(mainGraph.getTensors().get("o1") ==
                testOp2Tw.outputs().unwrap().at(2));
  BOOST_REQUIRE(mainGraph.getTensors().get("o2") ==
                testOp2Tw.outputs().unwrap().at(3));
}
