// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_7

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <iostream>
#include <string>

#include "get_results.hpp"
#include "popart/half.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_7_float_16,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // as test case 2, but with gradient accumulation

  float mm = 1.0f;
  float vs = 14.5f;
  float ls = 0.15f;

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, false}},
          {"defaultMomentum", {mm, false}},
          {"defaultVelocityScaling", {vs, false}},
          {"lossScaling", {ls, false}},
      },
      {},
      TestConfig::sgdAccMm);

  float wd = opt0.getUnsetWeightDecay().val();
  float dp = opt0.getUnsetDampening().val();

  auto opt1 = opt0;
  auto opt2 = opt0;

  float w0star = 100;
  float g0star = 0;
  float v0star = TestConfig::getInitialV(dp, wd, w0star);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true);

  float w1star = 200;
  float g1star = 0;
  float v1star = TestConfig::getInitialV(dp, wd, w1star);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true);

  auto results = getResults<popart::float16_t>(
      opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);

  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);

  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_7_float,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // as test case 2, but with gradient accumulation

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
  auto opt1 = opt0;
  auto opt2 = opt0;
  auto results =
      getResults<float>(opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);

  // Instead of emulating lagged updates, we pre-calculate the expected value.
  // Also, as the scalars are not changing over time, SGD1 and SGD2 give the
  // exact same results.

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

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_7_float_16_nesterov,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // Test nesterov momentum

  // as test case 2, but with gradient accumulation

  float mm = 1.0f;
  float vs = 14.5f;
  float ls = 0.15f;

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, false}},
          {"defaultMomentum", {mm, false}},
          {"defaultVelocityScaling", {vs, false}},
          {"lossScaling", {ls, false}},
          {"nesterov", {true, true}},
      },
      {},
      TestConfig::sgdAccMm);

  float wd = opt0.getUnsetWeightDecay().val();
  float dp = opt0.getUnsetDampening().val();

  auto opt1 = opt0;
  auto opt2 = opt0;

  float w0star = 100;
  float g0star = 0;
  float v0star = TestConfig::getInitialV(dp, wd, w0star);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true, true);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true, true);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, wd, mm, dp, lrTest0, false, true, true);

  float w1star = 200;
  float g1star = 0;
  float v1star = TestConfig::getInitialV(dp, wd, w1star);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true, true);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true, true);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, wd, mm, dp, lrTest0, false, true, true);

  auto results = getResults<popart::float16_t>(
      opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);

  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);

  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_7_float_nesterov,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // as test case 2, but with gradient accumulation

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, false}},
          {"defaultMomentum", {1.0f, false}},
          {"defaultVelocityScaling", {14.15f, false}},
          {"lossScaling", {0.15f, false}},
          {"nesterov", {true, true}},
      },
      {},
      TestConfig::sgdAccMm);
  auto opt1 = opt0;
  auto opt2 = opt0;
  auto results =
      getResults<float>(opt0, opt1, opt2, /*accl*/ true, /*repl*/ false);

  // Instead of emulating lagged updates, we pre-calculate the expected value.
  // Also, as the scalars are not changing over time, SGD1 and SGD2 give the
  // exact same results.

  // including factor 2 for accumulation factor
  auto absdiff0 =
      getAbsDiff(100 - accumulationFactor * 9 * lrTest0, std::get<0>(results));
  auto absdiff1 =
      getAbsDiff(200 - accumulationFactor * 9 * lrTest0, std::get<1>(results));
  std::cout << "abs diffs at float16: " << absdiff0 << " and " << absdiff1
            << std::endl;
  BOOST_CHECK(absdiff0 < 1e-5f);
  BOOST_CHECK(absdiff1 < 1e-5f);
}
