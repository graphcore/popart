// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_0

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <string>

#include "get_results.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_0,
                              TestConfig,
                              SGD1And2TestConfigs) {

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, true}},
          {"defaultMomentum", {1.0f, true}},
      },
      {},
      TestConfig::sgdAccMm);
  auto opt1 = opt0;
  auto opt2 = opt0;

  // using the pytorch equations for a weight W0:
  //
  // (1) g = g + wd * w
  // (2) v = v * mm + (1 - dp) * g
  // (3) w = w - lr * v

  // (t) g,v,w =  -1,0,W0
  // (1) g,v,w =  -1,0,W0
  // (2) g,v,w =  -1,-1, W0
  // (3) g,v,w =  -1,-1,W0-lr
  //
  // (t) g,v,w =  -1,-1,W0-lr
  // (1) g,v,w =  -1,-1,W0-lr
  // (2) g,v,w =  -1,-2,W0-lr
  // (3) g,v,w =  -1,-2,W0-3*lr
  //
  // (t) g,v,w =  -1,-2,W0-3*lr
  // (1) g,v,w =  -1,-2,W0-3*lr
  // (2) g,v,w =  -1,-3,W0-3*lr
  // (3) g,v,w =  -1,-3,W0-6*lr

  auto results = getResults<float>(opt0, opt1, opt2, false, false);
  // check that expected values are the same as read-back values
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_0_nesterov,
                              TestConfig,
                              SGD1And2TestConfigs) {

  // test nesterov momentum

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {lrTest0, true}},
          {"defaultMomentum", {1.0f, true}},
          {"nesterov", {true, true}},
      },
      {},
      TestConfig::sgdAccMm);
  auto opt1 = opt0;
  auto opt2 = opt0;

  // using the pytorch equations for a weight W0:
  //
  // (1) g = g + wd * w
  // (2) v = v * mm + (1 - dp) * g
  // (3) g = v * mm + g
  // (4) w = w - lr * g

  // (t) g,v,w =  -1,0,W0
  // (1) g,v,w =  -1,0,W0
  // (2) g,v,w =  -1,-1, W0
  // (3) g,v,w =  -2,-1, W0
  // (4) g,v,w =  -2,-1,W0-2*lr
  //
  // (t) g,v,w =  -1,-1,W0-2*lr
  // (1) g,v,w =  -1,-1,W0-2*lr
  // (2) g,v,w =  -1,-2,W0-2*lr
  // (3) g,v,w =  -3,-2,W0-2*lr
  // (4) g,v,w =  -3,-2,W0-5*lr
  //
  // (t) g,v,w =  -1,-2,W0-5*lr
  // (1) g,v,w =  -1,-2,W0-5*lr
  // (2) g,v,w =  -1,-3,W0-5*lr
  // (3) g,v,w =  -4,-3,W0-5*lr
  // (4) g,v,w =  -4,-3,W0-9*lr

  auto results = getResults<float>(opt0, opt1, opt2, false, false);
  // check that expected values are the same as read-back values
  auto absdiff0 = getAbsDiff(100 - 9 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 9 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
