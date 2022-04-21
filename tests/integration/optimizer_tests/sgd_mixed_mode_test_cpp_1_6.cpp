// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_6

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_6,
                              TestConfig,
                              SGD1And2TestConfigs) {

  // As in test 4, but decouple the 2 weights by using insertSpecific

  // const weight decay, different on the 2 weights
  float defaultWd = 0.2f;
  float weight1Wd = 0.3f;
  // variable default momentum, different on the 2 weights
  float defaultMm0 = 0.9f;
  float defaultMm1 = 0.6f;
  float defaultMm2 = 0.5f;
  float weight0Mm  = 0.85f; // constant momentum for weight 0
  // constant dampening, the same on the 2 weights
  float dp = 0.2f;
  // variable learning rate, different on the 2 weights
  float defaultLr0 = 1.0;
  float defaultLr1 = 0.9;
  float defaultLr2 = 0.8;
  float weight0Lr0 = 0.7;
  float weight0Lr1 = 0.6;
  float weight0Lr2 = 0.4;
  // constant loss scaling
  float ls = 600.0f;
  // constant velocity scaling, different on the 2 weights
  float defaultVs = 0.1;
  float weight0Vs = 0.01;

  popart::SGD opt0({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr0, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm0, false}}},
                   {},
                   TestConfig::sgdAccMm);

  // all values without a key in insertSpecific will take default values above
  opt0.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr0, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt0.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  popart::SGD opt1({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr1, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm1, false}}},
                   {},
                   TestConfig::sgdAccMm);
  opt1.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr1, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt1.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  popart::SGD opt2({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr2, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm2, false}}},
                   {},
                   TestConfig::sgdAccMm);
  opt2.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr2, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt2.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  float w0star = 100;
  float g0star = 0;
  float v0star = TestConfig::getInitialV(dp, defaultWd, w0star);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr0);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr1);
  TestConfig::laggedPytorchUpdate(
      w0star, g0star, v0star, defaultWd, weight0Mm, dp, weight0Lr2);

  float w1star = 200;
  float g1star = 0;
  float v1star = TestConfig::getInitialV(dp, weight1Wd, w1star);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm0, dp, defaultLr0);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm1, dp, defaultLr1);
  TestConfig::laggedPytorchUpdate(
      w1star, g1star, v1star, weight1Wd, defaultMm2, dp, defaultLr2);

  auto results  = getResults<float>(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
