// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TiedGatherTests
#include <boost/test/unit_test.hpp>

#include <typeindex>

#include <popart/patterns/tiedgatherpattern.hpp>

#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsTiedGather) {
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<TiedGatherPattern>());
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<TiedGatherAccumulatePattern>());
}

BOOST_AUTO_TEST_CASE(TestPatternsEnabledDisabledApiWorks) {
  Patterns ps;

  // Off by default.

  BOOST_REQUIRE(!ps.isTiedGatherEnabled());
  BOOST_REQUIRE(!ps.isTiedGatherAccumulateEnabled());

  // Calling enable works correctly.

  ps.enableTiedGather(true);
  ps.enableTiedGatherAccumulate(true);

  BOOST_REQUIRE(ps.isTiedGatherEnabled());
  BOOST_REQUIRE(ps.isTiedGatherAccumulateEnabled());

  // Calling disable through other api works correctly.

  ps.enablePattern(std::type_index(typeid(TiedGatherPattern)), false);
  ps.enablePattern("TiedGatherAccumulate", false);

  BOOST_REQUIRE(!ps.isPatternEnabled("TiedGather"));
  BOOST_REQUIRE(!ps.isPatternEnabled(
      std::type_index(typeid(TiedGatherAccumulatePattern))));
}
