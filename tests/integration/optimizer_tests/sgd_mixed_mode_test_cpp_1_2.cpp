// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_2

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <string>

#include "get_results.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_2,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // as test case 1, but non-const for all

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, false}},
          {"defaultMomentum", {1.0f, false}},
          {"defaultVelocityScaling", {14.15f, false}},
          {"lossScaling", {0.15f, false}},
      },
      {},
      TestConfig::sgdAccMm);
  auto opt1     = opt0;
  auto opt2     = opt0;
  auto results  = getResults<float>(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
