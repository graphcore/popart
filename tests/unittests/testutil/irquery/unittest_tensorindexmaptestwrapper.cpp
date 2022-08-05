// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_tensorindexmaptestwrapper

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#undef private

#include <map>
#include <memory>
#include <string>

#include <type_traits>

#include "mock_testfailuretriggerer.hpp"
#include "popart/graphid.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/optional.hpp"
#include "testutil/irquery/irquery.hpp"

namespace popart {
class Tensor;
} // namespace popart

using namespace popart;
using namespace popart::irquery;

namespace {

auto _ = trompeloeil::_;

/**
 * Subclass TensorIndexMapTestWrapper so we can replace triggerer with a mock
 * object. This allows us to test when it's triggered.
 */
class TestTensorIndexMapTestWrapper : public TensorIndexMapTestWrapper {
public:
  TestTensorIndexMapTestWrapper(Ir &ir,
                                const std::map<int, Tensor *> &tensorIndexMap,
                                const std::string &srcObjDescr,
                                const std::string &mapTypeDescrSingular,
                                const std::string &mapTypeDescrPlural)
      : TensorIndexMapTestWrapper{ir,
                                  tensorIndexMap,
                                  srcObjDescr,
                                  mapTypeDescrSingular,
                                  mapTypeDescrPlural} {
    // Overwrite triggerer with a mock triggerer.
    triggerer = std::make_unique<MockTestFailureTriggerer>();
  }

  MockTestFailureTriggerer *getMockTriggerer() {
    return static_cast<MockTestFailureTriggerer *>(&*triggerer);
  }
};
} // namespace

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_hasId) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TestTensorIndexMapTestWrapper tw{ir,
                                   std::map<int, Tensor *>{{1, t0}},
                                   "100 (TestOps.TestOp:1)",
                                   "input",
                                   "inputs"};

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasId("t0");
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasId("some_wrong_id"));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasId("t0", Require::MustBeTrue);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected 100 (TestOps.TestOp:1) to have input with "
                         "ID 'some_wrong_id' at some index (got inputs: "
                         "'t0')"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasId("some_wrong_id", Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect 100 (TestOps.TestOp:1) to have "
                         "input with ID 't0' at any index (got 't0' at "
                         "index 1)"))
        .TIMES(1);
    auto result = tw.hasId("t0", Require::MustBeFalse);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasId("some_wrong_id", Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_hasIndex) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TestTensorIndexMapTestWrapper tw{ir,
                                   std::map<int, Tensor *>{{1, t0}},
                                   "100 (TestOps.TestOp:1)",
                                   "input",
                                   "inputs"};

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasIndex(1);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIndex(0));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasIndex(1, Require::MustBeTrue);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected 100 (TestOps.TestOp:1) to have input at "
                         "index 2 but input is not connected"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasIndex(2, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect 100 (TestOps.TestOp:1) to have input "
                         "at index 1 (got 't0' at index 1)"))
        .TIMES(1);
    auto result = tw.hasIndex(1, Require::MustBeFalse);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIndex(2, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_hasIdAtIndex) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TestTensorIndexMapTestWrapper tw{ir,
                                   std::map<int, Tensor *>{{1, t0}},
                                   "100 (TestOps.TestOp:1)",
                                   "input",
                                   "inputs"};

  // No triggering -- Require::Nothing.

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct index + id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasIdAtIndex(1, "t0");
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIdAtIndex(0, "t0"));
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIdAtIndex(1, "t1"));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    auto result = tw.hasIdAtIndex(1, "t0", Require::MustBeTrue);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected 100 (TestOps.TestOp:1) to have input at "
                         "index 2 with ID 't0' (input is not connected)"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasIdAtIndex(2, "t0", Require::MustBeTrue));
  }
  {
    // Wrong id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected 100 (TestOps.TestOp:1) to have input at "
                         "index 1 with ID 'h6' (got 't0' at index 1)"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasIdAtIndex(1, "h6", Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct index + id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect 100 (TestOps.TestOp:1) to have input "
                         "at index 1 with ID 't0'"))
        .TIMES(1);
    auto result = tw.hasIdAtIndex(1, "t0", Require::MustBeFalse);
    BOOST_REQUIRE(result);
    BOOST_REQUIRE(1 == result->index());
    BOOST_REQUIRE(t0 == result->tensor().unwrap());
  }
  {
    // Wrong index.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIdAtIndex(2, "t0", Require::MustBeFalse));
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasIdAtIndex(1, "h6", Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_containsIds) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  ir.getMainGraph().getTensors().addActGrad("a9");
  auto a9 = ir.getMainGraph().getTensors().get("a9");
  TestTensorIndexMapTestWrapper tw{ir,
                                   std::map<int, Tensor *>{{1, t0}, {2, a9}},
                                   "100 (TestOps.TestOp:1)",
                                   "input",
                                   "inputs"};

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.containsIds({"t0", "a9"}));
  }
  {
    // Subset of inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.containsIds({"a9"}));
  }
  {
    // Wrong inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.containsIds({"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.containsIds({"t0", "a9"}, Require::MustBeTrue));
  }
  {
    // Subset of inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.containsIds({"a9"}, Require::MustBeTrue));
  }
  {
    // Wrong inputs.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected 100 (TestOps.TestOp:1)'s inputs to "
                         "include {'a9', 'f4'} but got {'t0', 'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.containsIds({"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect 100 (TestOps.TestOp:1)'s inputs to "
                         "include {'t0', 'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(tw.containsIds({"t0", "a9"}, Require::MustBeFalse));
  }
  {
    // Subset of inputs.
    REQUIRE_CALL(
        *tw.getMockTriggerer(),
        trigger(
            "Did not expect 100 (TestOps.TestOp:1)'s inputs to include {'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(tw.containsIds({"a9"}, Require::MustBeFalse));
  }
  {
    // Wrong inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.containsIds({"a9", "f4"}, Require::MustBeFalse));
  }
}

BOOST_AUTO_TEST_CASE(tensorindexmaptestwrapper_hasExactIds) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  ir.getMainGraph().getTensors().addActGrad("a9");
  auto a9 = ir.getMainGraph().getTensors().get("a9");
  TestTensorIndexMapTestWrapper tw{ir,
                                   std::map<int, Tensor *>{{1, t0}, {2, a9}},
                                   "100 (TestOps.TestOp:1)",
                                   "input",
                                   "inputs"};

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Exact inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasExactIds({"t0", "a9"}));
  }
  {
    // Subset of inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasExactIds({"a9"}));
  }
  {
    // Wrong inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasExactIds({"a9", "f4"}));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasExactIds({"t0", "a9"}, Require::MustBeTrue));
  }
  {
    // Subset of inputs.
    REQUIRE_CALL(
        *tw.getMockTriggerer(),
        trigger("Expected 100 (TestOps.TestOp:1)'s inputs to be {'a9'} "
                "but got {'t0', 'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasExactIds({"a9"}, Require::MustBeTrue));
  }
  {
    // Wrong inputs.
    REQUIRE_CALL(
        *tw.getMockTriggerer(),
        trigger("Expected 100 (TestOps.TestOp:1)'s inputs to be {'a9', "
                "'f4'} but got {'t0', 'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasExactIds({"a9", "f4"}, Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Exact inputs.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect 100 (TestOps.TestOp:1)'s inputs to be "
                         "{'t0', 'a9'}"))
        .TIMES(1);
    BOOST_REQUIRE(tw.hasExactIds({"t0", "a9"}, Require::MustBeFalse));
  }
  {
    // Subset of inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasExactIds({"a9"}, Require::MustBeFalse));
  }
  {
    // Wrong inputs.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasExactIds({"a9", "f4"}, Require::MustBeFalse));
  }
}
