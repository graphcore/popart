// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GenOperatorsUnittest

#include <boost/test/unit_test.hpp>
#include <vector>

#include "popart/builder.hpp"
#include "popart/tensorinfo.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(TestUniformRandom) {
  auto builder = Builder::create();

  auto aiOnnx = builder->aiOnnxOpset6();

  const float expectedLow  = std::numeric_limits<float>::min();
  const float expectedHigh = 1.0 - std::numeric_limits<float>::epsilon();

  auto tensor = aiOnnx.randomuniform(
      std::vector<int64_t>{2, 2}, 1, expectedHigh, expectedLow);

  bool hasLow = builder->nodeHasAttribute("low", {tensor});
  BOOST_TEST(hasLow);

  if (hasLow) {
    auto low = builder->getFloatNodeAttribute("low", {tensor});
    BOOST_TEST(expectedLow == low);
  }

  bool hasHigh = builder->nodeHasAttribute("high", {tensor});
  BOOST_TEST(hasHigh);

  if (hasHigh) {
    auto high = builder->getFloatNodeAttribute("high", {tensor});
    BOOST_TEST(expectedHigh == high);
  }
}
