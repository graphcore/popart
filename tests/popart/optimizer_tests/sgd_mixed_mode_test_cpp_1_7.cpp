// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_7

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_7_float_16) {
  // as test case 2, but with gradient accumulation

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, false}},
      {"defaultMomentum", {1.0f, false}},
      {"defaultVelocityScaling", {14.15f, false}},
      {"lossScaling", {0.15f, false}},
  });
  auto opt1    = opt0;
  auto opt2    = opt0;
  auto results = getResults<popart::float16_t>(
      opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);
  // including factor 2 for accumulation factor
  auto absdiff0 =
      getAbsDiff(100 - accumulationFactor * 6 * lrTest0, std::get<0>(results));
  auto absdiff1 =
      getAbsDiff(200 - accumulationFactor * 6 * lrTest0, std::get<1>(results));
  std::cout << "abs diffs at float32: " << absdiff0 << " and " << absdiff1
            << std::endl;
  BOOST_CHECK(absdiff0 < 1e-5f);
  BOOST_CHECK(absdiff1 < 1e-5f);
}

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_7_float) {
  // as test case 2, but with gradient accumulation

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, false}},
      {"defaultMomentum", {1.0f, false}},
      {"defaultVelocityScaling", {14.15f, false}},
      {"lossScaling", {0.15f, false}},
  });
  auto opt1 = opt0;
  auto opt2 = opt0;
  auto results =
      getResults<float>(opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);
  // including factor 2 for accumulation factor
  auto absdiff0 =
      getAbsDiff(100 - accumulationFactor * 6 * lrTest0, std::get<0>(results));
  auto absdiff1 =
      getAbsDiff(200 - accumulationFactor * 6 * lrTest0, std::get<1>(results));
  std::cout << "abs diffs at float16: " << absdiff0 << " and " << absdiff1
            << std::endl;
  BOOST_CHECK(absdiff0 < 1e-5f);
  BOOST_CHECK(absdiff1 < 1e-5f);
}
