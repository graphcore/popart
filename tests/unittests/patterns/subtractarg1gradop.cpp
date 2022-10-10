// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TestPatternsSubtractArg1GradOp
#include <boost/test/unit_test.hpp>

#include <popart/patterns/patterns.hpp>
#include <popart/patterns/subtractarg1gradoppattern.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPatternEnabledAlways) {
  const auto testEnabled = [](Patterns &ps) {
    BOOST_REQUIRE(ps.isSubtractArg1GradOpEnabled());
    BOOST_REQUIRE(ps.isPatternEnabled(typeid(SubtractArg1GradOpPattern)));
    BOOST_REQUIRE(ps.isPatternEnabled("SubtractArg1GradOp"));
  };

  {
    Patterns minimal_patterns(PatternsLevel::Minimal);
    testEnabled(minimal_patterns);
  }
  {
    Patterns default_patterns;
    testEnabled(default_patterns);
  }
}
