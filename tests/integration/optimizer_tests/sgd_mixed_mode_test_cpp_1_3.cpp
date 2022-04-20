// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_3

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <string>

#include "get_results.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_3,
                              TestConfig,
                              SGD1And2TestConfigs) {

  // Caveat about updating SGD and comparing to pytorch:
  // ---------------------------------
  // Pytorch effectively does:
  //
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  //    v <- v * mm  + (1 - dp) * wd * w
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  // x
  // ...
  //
  // Our implementation is the same, but the optimizer value updates do not
  // come at the x's. We do:
  //
  // Host side initialization with :
  //    v <- (1 - dp) * wd * w,
  //
  //  and then we repeat:
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //    v <- v + (1 - dp) * g
  //    w <- w - lr * v
  //    v <- v * mm  + (1 - dp) * wd * w
  // y
  //
  // The update of optimizer parameters takes place at the 'y's.
  //
  // So there is a lag, which will be noticed if mm, dp, and wd are updated;
  // but not if lr is.

  float wd0 = 0.1;
  float mm0 = 0.3;
  float dp0 = 0.4;
  float lr0 = 0.2;

  float wd1 = 0.1;
  float mm1 = 0.3;
  float dp1 = 0.4;
  float lr1 = 0.2; // learning rate can change and still be identical to pytorch

  float wd2 = 0.1;
  float mm2 = 0.3;
  float dp2 = 0.4;
  float lr2 = 0.3;

  popart::SGD opt0({{"defaultDampening", {dp0, true}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, true}},
                    {"defaultMomentum", {mm0, false}}},
                   {},
                   TestConfig::sgdAccMm);

  popart::SGD opt1({{"defaultDampening", {dp1, true}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, true}},
                    {"defaultMomentum", {mm1, false}}},
                   {},
                   TestConfig::sgdAccMm);

  popart::SGD opt2({{"defaultDampening", {dp2, true}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, true}},
                    {"defaultMomentum", {mm2, false}}},
                   {},
                   TestConfig::sgdAccMm);

  float w0star{100};
  float g0star{0};
  float v0star{0};
  pytorchUpdate(w0star, g0star, v0star, wd0, mm0, dp0, lr0);
  pytorchUpdate(w0star, g0star, v0star, wd1, mm1, dp1, lr1);
  pytorchUpdate(w0star, g0star, v0star, wd2, mm2, dp2, lr2);

  float w1star{200};
  float g1star{0};
  float v1star{0};
  pytorchUpdate(w1star, g1star, v1star, wd0, mm0, dp0, lr0);
  pytorchUpdate(w1star, g1star, v1star, wd1, mm1, dp1, lr1);
  pytorchUpdate(w1star, g1star, v1star, wd2, mm2, dp2, lr2);

  auto results  = getResults<float>(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
