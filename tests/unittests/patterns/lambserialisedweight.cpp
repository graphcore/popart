// Copyright(c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE LambSerialisedWeightTests
#include <boost/test/unit_test.hpp>

#include <popart/patterns/lambserialisedweight.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/lamb.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

#include <typeindex>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPatternNamesContainsLambSerialisedWeight) {
  BOOST_REQUIRE_NO_THROW(PatternNames::getName<LambSerialisedWeightPattern>());
}

BOOST_AUTO_TEST_CASE(TestPatternsEnabledDisabledApiWorks) {
  Patterns ps;

  // On by default.
  BOOST_REQUIRE(ps.isLambSerialisedWeightEnabled());

  // Calling disable works correctly.
  ps.enablePattern(std::type_index(typeid(LambSerialisedWeightPattern)), false);
  BOOST_REQUIRE(!ps.isPatternEnabled("LambSerialisedWeight"));

  // Calling enable (through another api) works correctly.
  ps.enablePattern("LambSerialisedWeight", true);
  BOOST_REQUIRE(ps.isPatternEnabled(
      std::type_index(typeid(LambSerialisedWeightPattern))));

  // Calling disable (through another api) works correctly.
  ps.enableLambSerialisedWeight(false);
  BOOST_REQUIRE(!ps.isLambSerialisedWeightEnabled());
}
