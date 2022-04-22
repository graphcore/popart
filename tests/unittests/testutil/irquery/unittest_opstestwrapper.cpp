// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_opstestwrapper

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#undef private

#include <testutil/irquery/mock_testfailuretriggerer.hpp>
#include <testutil/irquery/testop.hpp>
#include <type_traits>
#include <vector>

#include "popart/datatype.hpp"
#include "popart/graphid.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/optional.hpp"
#include "testutil/irquery/irquery.hpp"

using namespace popart;
using namespace popart::irquery;

namespace {

auto _ = trompeloeil::_;

/**
 * Subclass OpsTestWrapper so we can replace triggerer with a mock object. This
 * allows us to test when it's triggered.
 */
class TestOpsTestWrapper : public OpsTestWrapper {
public:
  TestOpsTestWrapper(Ir &ir,
                     const std::vector<Op *> &ops,
                     const std::string &srcObjDescr)
      : OpsTestWrapper{ir, ops, srcObjDescr} {
    // Overwrite triggerer with a mock triggerer.
    triggerer = std::make_unique<MockTestFailureTriggerer>();
  }

  MockTestFailureTriggerer *getMockTriggerer() {
    return static_cast<MockTestFailureTriggerer *>(&*triggerer);
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(opstestwrapper_hasGraph) {

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

  TestOpsTestWrapper tw{ir, {testOp0, testOp1, testOp2}, "Some Op Source"};

  auto opPred0 = [&](auto &opTw) -> bool { return (true); };
  auto opPred1 = [&](auto &opTw) -> bool {
    return bool(opTw.inputs().hasIndex(3));
  };
  auto opPred2 = [&](auto &opTw) -> bool {
    return bool(opTw.outputs().hasIndex(17));
  };

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.
  {
    // There is a TestOp (there is, could be any of them).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasOp<TestOp>(opPred0));
  }
  {
    // There is a TestOp with an input on index 3 (there is, testOp2).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(bool(tw.hasOp<TestOp>(opPred1)));
    BOOST_REQUIRE(testOp2 == (*tw.hasOp<TestOp>(opPred1)).unwrap());
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasOp<TestOp>(opPred2));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // There is a TestOp (there is, could be any of them).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasOp<TestOp>(opPred0, Require::MustBeTrue));
  }
  {
    // There is a TestOp with an input on index 3 (there is, testOp2).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasOp<TestOp>(opPred1, Require::MustBeTrue));
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected to find an op in "
                         "Some Op Source that matches "
                         "predicate"));
    BOOST_REQUIRE(!tw.hasOp<TestOp>(opPred2, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // There is a TestOp (there is, could be any of them).
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect to find an op "
                         "in Some Op Source that "
                         "matches predicate"));
    BOOST_REQUIRE(tw.hasOp<TestOp>(opPred0, Require::MustBeFalse));
  }
  {
    // There is a TestOp with an input on index 3 (there is, testOp2).
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect to find an op "
                         "in Some Op Source that "
                         "matches predicate"));
    BOOST_REQUIRE(tw.hasOp<TestOp>(opPred1, Require::MustBeFalse));
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasOp<TestOp>(opPred2, Require::MustBeFalse));
  }
}
