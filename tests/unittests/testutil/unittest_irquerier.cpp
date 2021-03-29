// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE autodiff_unittest

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>

#include <iostream>
#include <sstream>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/transforms/autodiff.hpp>

#include <testutil/irquery/irquerier.hpp>

using namespace popart;
using namespace popart::testing;

namespace {

auto _ = trompeloeil::_;

/**
 * A simple test op class for use in tests in this file.
 **/
class TestOp : public Op {
public:
  TestOp(const Op::Settings &settings)
      : Op(OperatorIdentifier("TestOps", "TestOp", 1), settings) {}

  void setup() override {
    for (const auto &entry : input->tensorIdMap()) {
      outInfo(entry.first) = inInfo(entry.first);
    }
  }

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<TestOp>(*this);
  }

  virtual float getSubgraphValue() const override {
    return getLowSubgraphValue();
  }
};

/**
 * We mock the 'triggerTestFailure' function of IrQuerier because we want to
 * test if BOOST_REQUIRE is called at the right times. This isn't really a
 * conventional way of using mocks but it's convenient. If need be we can break
 * out the call to 'triggerTestFailure' in a separate interface dependency.
 */
class TestIrQuerier : public IrQuerier {
  MAKE_MOCK1(triggerTestFailure, void(std::string), override);
};

} // namespace

BOOST_AUTO_TEST_CASE(irquery_graphHasOp) {

  TestIrQuerier q;
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

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.
  {
    // There is a TestOp (there is).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.graphHasOp(mainGraph, [](Op *op) -> bool {
      return (dynamic_cast<TestOp *>(op) != nullptr);
    }));
  }
  {
    // There is a TestOp with an input on index 3 (there is).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.graphHasOp(mainGraph, [&q](Op *op) -> bool {
      return (dynamic_cast<TestOp *>(op) != nullptr) && (q.opHasInputAt(op, 3));
    }));
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.graphHasOp(mainGraph, [&q](Op *op) -> bool {
      return (dynamic_cast<TestOp *>(op) != nullptr) &&
             (q.opHasOutputAt(op, 17));
    }));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // There is a TestOp (there is).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.graphHasOp(
        mainGraph,
        [](Op *op) -> bool { return (dynamic_cast<TestOp *>(op) != nullptr); },
        Require::MustBeTrue));
  }
  {
    // There is a TestOp with an input on index 3 (there is).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.graphHasOp(
        mainGraph,
        [&q](Op *op) -> bool {
          return (dynamic_cast<TestOp *>(op) != nullptr) &&
                 (q.opHasInputAt(op, 3));
        },
        Require::MustBeTrue));
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    REQUIRE_CALL(
        q,
        triggerTestFailure(
            "Expected to find an op in the main graph that matches predicate"));
    BOOST_REQUIRE(!q.graphHasOp(
        mainGraph,
        [&q](Op *op) -> bool {
          return (dynamic_cast<TestOp *>(op) != nullptr) &&
                 (q.opHasOutputAt(op, 17));
        },
        Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // There is a TestOp (there is).
    REQUIRE_CALL(q,
                 triggerTestFailure("Did not expect to find an op in the main "
                                    "graph that matches predicate"));
    BOOST_REQUIRE(q.graphHasOp(
        mainGraph,
        [](Op *op) -> bool { return (dynamic_cast<TestOp *>(op) != nullptr); },
        Require::MustBeFalse));
  }
  {
    // There is a TestOp with an input on index 3 (there is).
    REQUIRE_CALL(q,
                 triggerTestFailure("Did not expect to find an op in the main "
                                    "graph that matches predicate"));
    BOOST_REQUIRE(q.graphHasOp(
        mainGraph,
        [&q](Op *op) -> bool {
          return (dynamic_cast<TestOp *>(op) != nullptr) &&
                 (q.opHasInputAt(op, 3));
        },
        Require::MustBeFalse));
  }
  {
    // There is a TestOp with an output on index 17 (there isn't).
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.graphHasOp(
        mainGraph,
        [&q](Op *op) -> bool {
          return (dynamic_cast<TestOp *>(op) != nullptr) &&
                 (q.opHasOutputAt(op, 17));
        },
        Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasInputAt) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputAt(testOp, 1));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputAt(testOp, 0));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputAt(testOp, 1, Require::MustBeTrue));
  }
  {
    // Wrong index.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have input at "
                           "index 2 but input is not connected"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasInputAt(testOp, 2, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Did not expect 100 (TestOps.TestOp:1) to have input "
                     "at index 1 ('t0')"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasInputAt(testOp, 1, Require::MustBeFalse));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputAt(testOp, 2, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasInputIdAt) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->setup();

  // No triggering -- Require::Nothing.

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index + id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIdAt(testOp, 1, "t0"));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 0, "t0"));
  }
  {
    // Wrong id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 1, "t1"));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIdAt(testOp, 1, "t0", Require::MustBeTrue));
  }
  {
    // Wrong index.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have input at "
                           "index 2 with ID 't0' but input is not connected"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 2, "t0", Require::MustBeTrue));
  }
  {
    // Wrong id.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have input at "
                           "index 1 with ID 'h6' but got ID 't0'"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 1, "h6", Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Did not expect 100 (TestOps.TestOp:1) to have input "
                     "at index 1 with ID 't0'"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasInputIdAt(testOp, 1, "t0", Require::MustBeFalse));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 2, "t0", Require::MustBeFalse));
  }
  {
    // Wrong id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIdAt(testOp, 1, "h6", Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasInputIds) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("a9", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->connectInTensor(2, "a9");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->createAndConnectOutTensor(2, "p6");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"t0", "a9"}));
  }
  {
    // Subset of inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"a9"}));
  }
  {
    // Wrong inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIds(testOp, {"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"t0", "a9"}, Require::MustBeTrue));
  }
  {
    // Subset of inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"a9"}, Require::MustBeTrue));
  }
  {
    // Wrong inputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to include inputs "
                           "{a9, f4} but got {t0, a9}"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasInputIds(testOp, {"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Did not expect 100 (TestOps.TestOp:1)'s inputs to "
                           "include {a9, t0}"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"t0", "a9"}, Require::MustBeFalse));
  }
  {
    // Subset of inputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure(
            "Did not expect 100 (TestOps.TestOp:1)'s inputs to include {a9}"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasInputIds(testOp, {"a9"}, Require::MustBeFalse));
  }
  {
    // Wrong inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasInputIds(testOp, {"a9", "f4"}, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasExactInputIds) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("a9", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->connectInTensor(2, "a9");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->createAndConnectOutTensor(2, "p6");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasExactInputIds(testOp, {"t0", "a9"}));
  }
  {
    // Subset of inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactInputIds(testOp, {"a9"}));
  }
  {
    // Wrong inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactInputIds(testOp, {"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(
        q.opHasExactInputIds(testOp, {"t0", "a9"}, Require::MustBeTrue));
  }
  {
    // Subset of inputs.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Expected 100 (TestOps.TestOp:1) to have inputs {a9} "
                     "but got {t0, a9}"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasExactInputIds(testOp, {"a9"}, Require::MustBeTrue));
  }
  {
    // Wrong inputs.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Expected 100 (TestOps.TestOp:1) to have inputs {a9, "
                     "f4} but got {t0, a9}"))
        .TIMES(1);
    BOOST_REQUIRE(
        !q.opHasExactInputIds(testOp, {"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure(
            "Did not expect 100 (TestOps.TestOp:1) to have inputs {t0, a9}"))
        .TIMES(1);
    BOOST_REQUIRE(
        q.opHasExactInputIds(testOp, {"t0", "a9"}, Require::MustBeFalse));
  }
  {
    // Subset of inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactInputIds(testOp, {"a9"}, Require::MustBeFalse));
  }
  {
    // Wrong inputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(
        !q.opHasExactInputIds(testOp, {"a9", "f4"}, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasOutputAt) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputAt(testOp, 1));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputAt(testOp, 0));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputAt(testOp, 1, Require::MustBeTrue));
  }
  {
    // Wrong index.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have output at "
                           "index 2 but output is not connected"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasOutputAt(testOp, 2, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Did not expect 100 (TestOps.TestOp:1) to have output "
                     "at index 1 ('o1')"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasOutputAt(testOp, 1, Require::MustBeFalse));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputAt(testOp, 2, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasOutputIdAt) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->setup();

  // No triggering -- Require::Nothing.

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index + id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIdAt(testOp, 1, "o1"));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 0, "t0"));
  }
  {
    // Wrong id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 1, "t1"));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIdAt(testOp, 1, "o1", Require::MustBeTrue));
  }
  {
    // Wrong index.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have output at "
                           "index 2 with ID 't0' but output is not connected"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 2, "t0", Require::MustBeTrue));
  }
  {
    // Wrong id.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to have output at "
                           "index 1 with ID 'h6' but got ID 'o1'"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 1, "h6", Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Did not expect 100 (TestOps.TestOp:1) to have output "
                     "at index 1 with ID 'o1'"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasOutputIdAt(testOp, 1, "o1", Require::MustBeFalse));
  }
  {
    // Wrong index.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 2, "t0", Require::MustBeFalse));
  }
  {
    // Wrong id.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputIdAt(testOp, 1, "h6", Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasOutputIds) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("a9", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->connectInTensor(2, "a9");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->createAndConnectOutTensor(2, "p6");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"o1", "p6"}));
  }
  {
    // Subset of outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"p6"}));
  }
  {
    // Wrong outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasOutputIds(testOp, {"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"o1", "p6"}, Require::MustBeTrue));
  }
  {
    // Subset of outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"p6"}, Require::MustBeTrue));
  }
  {
    // Wrong outputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Expected 100 (TestOps.TestOp:1) to include outputs "
                           "{a9, f4} but got {o1, p6}"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasOutputIds(testOp, {"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact outputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure("Did not expect 100 (TestOps.TestOp:1)'s outputs to "
                           "include {o1, p6}"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"o1", "p6"}, Require::MustBeFalse));
  }
  {
    // Subset of outputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure(
            "Did not expect 100 (TestOps.TestOp:1)'s outputs to include {p6}"))
        .TIMES(1);
    BOOST_REQUIRE(q.opHasOutputIds(testOp, {"p6"}, Require::MustBeFalse));
  }
  {
    // Wrong outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(
        !q.opHasOutputIds(testOp, {"a9", "f4"}, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(irquery_opHasExactOutputIds) {

  TestIrQuerier q;
  Ir ir;
  auto &mainGraph       = ir.getMainGraph();
  Op::Settings settings = Op::Settings{mainGraph, "SimpleTestOp"};

  // Add "t0" to the main graph.
  TensorInfo t0Info{DataType::INT32, {}};
  int32_t t0Data[] = {5};
  mainGraph.getTensors().addVarInit("t0", t0Info, static_cast<void *>(&t0Data));
  mainGraph.getTensors().addVarInit("a9", t0Info, static_cast<void *>(&t0Data));

  // Add SimpleTestOp.
  auto testOp = mainGraph.createOp<TestOp>(settings);

  // Connect "t0" to SimpleTestOp's input, create the "t1" output and call
  // setup().
  testOp->connectInTensor(1, "t0");
  testOp->connectInTensor(2, "a9");
  testOp->createAndConnectOutTensor(1, "o1");
  testOp->createAndConnectOutTensor(2, "p6");
  testOp->setup();

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(q.opHasExactOutputIds(testOp, {"o1", "p6"}));
  }
  {
    // Subset of outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactOutputIds(testOp, {"p6"}));
  }
  {
    // Wrong outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactOutputIds(testOp, {"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(
        q.opHasExactOutputIds(testOp, {"o1", "p6"}, Require::MustBeTrue));
  }
  {
    // Subset of outputs.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Expected 100 (TestOps.TestOp:1) to have outputs {p6} "
                     "but got {o1, p6}"))
        .TIMES(1);
    BOOST_REQUIRE(!q.opHasExactOutputIds(testOp, {"p6"}, Require::MustBeTrue));
  }
  {
    // Wrong outputs.
    REQUIRE_CALL(q,
                 triggerTestFailure(
                     "Expected 100 (TestOps.TestOp:1) to have outputs {a9, "
                     "f4} but got {o1, p6}"))
        .TIMES(1);
    BOOST_REQUIRE(
        !q.opHasExactOutputIds(testOp, {"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact outputs.
    REQUIRE_CALL(
        q,
        triggerTestFailure(
            "Did not expect 100 (TestOps.TestOp:1) to have outputs {o1, p6}"))
        .TIMES(1);
    BOOST_REQUIRE(
        q.opHasExactOutputIds(testOp, {"o1", "p6"}, Require::MustBeFalse));
  }
  {
    // Subset of outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(!q.opHasExactOutputIds(testOp, {"p6"}, Require::MustBeFalse));
  }
  {
    // Wrong outputs.
    FORBID_CALL(q, triggerTestFailure(_));
    BOOST_REQUIRE(
        !q.opHasExactOutputIds(testOp, {"a9", "f4"}, Require::MustBeFalse));
  }
}
