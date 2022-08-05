// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_irtestwrapper

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#undef private

#include <functional>
#include <memory>
#include <type_traits>

#include "mock_testfailuretriggerer.hpp"
#include "popart/graphid.hpp"
#include "popart/vendored/optional.hpp"
#include "testutil/irquery/irquery.hpp"

using namespace popart;
using namespace popart::irquery;

namespace {

auto _ = trompeloeil::_;

/**
 * Subclass IrTestWrapper so we can replace triggerer with a mock object. This
 * allows us to test when it's triggered.
 */
class TestIrTestWrapper : public IrTestWrapper {
public:
  TestIrTestWrapper(Ir &ir) : IrTestWrapper{ir} {
    // Overwrite triggerer with a mock triggerer.
    triggerer = std::make_unique<MockTestFailureTriggerer>();
  }

  MockTestFailureTriggerer *getMockTriggerer() {
    return static_cast<MockTestFailureTriggerer *>(&*triggerer);
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(irtestwrapper_hasGraph) {

  Ir ir;
  ir.createGraph(GraphId{"B"});
  TestIrTestWrapper tw{ir};

  // First, test all Require::Nothing cases. These should never result in
  // test failures but the result returned by the function should match
  // what we expect.

  {
    // Correct id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasGraph(GraphId("B")));
    BOOST_REQUIRE(GraphId("B") == tw.hasGraph(GraphId("B"))->unwrap().get().id);
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasGraph(GraphId("H")));
  }

  // Now, test all Require::MustBeTrue cases. These should result in a test
  // failure if the query is true. We also check the result returned by the
  // function match what we expect.
  {
    // Correct id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(tw.hasGraph(GraphId("B"), Require::MustBeTrue));
  }
  {
    // Wrong id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Expected to find graph with ID 'H' in IR"))
        .TIMES(1);
    BOOST_REQUIRE(!tw.hasGraph(GraphId("H"), Require::MustBeTrue));
  }

  // Finally, test all Require::MustBeFalse cases. These should result in a test
  // failure if the query is false. We also check the result returned by the
  // function match what we expect.
  {
    // Correct id.
    REQUIRE_CALL(*tw.getMockTriggerer(),
                 trigger("Did not expect to find graph with ID 'B' in IR"))
        .TIMES(1);
    BOOST_REQUIRE(tw.hasGraph(GraphId("B"), Require::MustBeFalse));
  }
  {
    // Wrong id.
    FORBID_CALL(*tw.getMockTriggerer(), trigger(_));
    BOOST_REQUIRE(!tw.hasGraph(GraphId("H"), Require::MustBeFalse));
  }
}
