// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_5

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <string>

#include "get_results.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_5,
                              TestConfig,
                              SGD1And2TestConfigs) {

  // As in test 4, but now with velocity scaling term changing.

  float wd0 = 0.4f;
  float mm0 = 0.6f;
  float dp0 = 0.1f;
  float lr0 = 0.2f;
  float ls0 = 10.0f;
  float vs0 = 3.0f;

  float wd1 = 0.15f;
  float mm1 = 0.35f;
  float dp1 = 0.45f;
  float lr1 = 0.55f;
  float ls1 = 8.0f;
  float vs1 = 6.0f;

  float wd2 = 0.03f;
  float mm2 = 0.83f;
  float dp2 = 0.13f;
  float lr2 = 0.25f;
  float ls2 = 6.0f;
  float vs2 = 9.0f;

  popart::SGD opt0({{"defaultDampening", {dp0, false}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, false}},
                    {"defaultVelocityScaling", {vs0, false}},
                    {"lossScaling", {ls0, false}},
                    {"defaultMomentum", {mm0, false}}},
                   {},
                   TestConfig::sgdAccMm);

  popart::SGD opt1({{"defaultDampening", {dp1, false}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, false}},
                    {"defaultVelocityScaling", {vs1, false}},
                    {"lossScaling", {ls1, false}},
                    {"defaultMomentum", {mm1, false}}},
                   {},
                   TestConfig::sgdAccMm);

  popart::SGD opt2({{"defaultDampening", {dp2, false}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, false}},
                    {"defaultVelocityScaling", {vs2, false}},
                    {"lossScaling", {ls2, false}},
                    {"defaultMomentum", {mm2, false}}},
                   {},
                   TestConfig::sgdAccMm);

  float w0star = 100;
  float g0star = 0;
  float v0star = TestConfig::getInitialV(dp0, wd0, vs0, w0star);
  TestConfig::laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd0, mm0, dp0, lr0, vs0, ls0);
  TestConfig::laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd1, mm1, dp1, lr1, vs1, ls1);
  TestConfig::laggedPytorchUpdateWithScaling(
      w0star, g0star, v0star, wd2, mm2, dp2, lr2, vs2, ls2);

  float w1star = 200;
  float g1star = 0;
  float v1star = TestConfig::getInitialV(dp0, wd0, vs0, w1star);
  TestConfig::laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd0, mm0, dp0, lr0, vs0, ls0);
  TestConfig::laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd1, mm1, dp1, lr1, vs1, ls1);
  TestConfig::laggedPytorchUpdateWithScaling(
      w1star, g1star, v1star, wd2, mm2, dp2, lr2, vs2, ls2);

  auto results  = getResults<float>(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
