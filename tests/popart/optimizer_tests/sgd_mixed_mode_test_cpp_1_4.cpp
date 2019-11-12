#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_4

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_4) {

  // Testing that results match pytorch with the caveat discussed in test 3.

  float wd0 = 0.4f;
  float mm0 = 0.6f;
  float dp0 = 0.1f;
  float lr0 = 0.2f;
  float ls0 = 10.0f;

  float wd1 = 0.15f;
  float mm1 = 0.35f;
  float dp1 = 0.45f;
  float lr1 = 0.55f;
  float ls1 = 8.0f;

  float wd2 = 0.03f;
  float mm2 = 0.83f;
  float dp2 = 0.13f;
  float lr2 = 0.25f;
  float ls2 = 6.0f;

  // Note: see test 5 for velocity scaling changing during training
  float vs = 3.0f;

  popart::SGD opt0({{"defaultDampening", {dp0, false}},
                    {"defaultLearningRate", {lr0, false}},
                    {"defaultWeightDecay", {wd0, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls0, false}},
                    {"defaultMomentum", {mm0, false}}});

  popart::SGD opt1({{"defaultDampening", {dp1, false}},
                    {"defaultLearningRate", {lr1, false}},
                    {"defaultWeightDecay", {wd1, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls1, false}},
                    {"defaultMomentum", {mm1, false}}});

  popart::SGD opt2({{"defaultDampening", {dp2, false}},
                    {"defaultLearningRate", {lr2, false}},
                    {"defaultWeightDecay", {wd2, false}},
                    {"defaultVelocityScaling", {vs, true}},
                    {"lossScaling", {ls2, false}},
                    {"defaultMomentum", {mm2, false}}});

  auto getInitialV = [](float dp, float wd, float W) {
    // note that we do not include the velocity scaling term here, pytorch
    // updates are independent of velocity scaling.
    return (1.0f - dp) * wd * W;
  };

  float w0star = 100;
  float g0star = 0;
  float v0star = getInitialV(dp0, wd0, w0star);
  laggedPytorchUpdate(w0star, g0star, v0star, wd0, mm0, dp0, lr0);
  laggedPytorchUpdate(w0star, g0star, v0star, wd1, mm1, dp1, lr1);
  laggedPytorchUpdate(w0star, g0star, v0star, wd2, mm2, dp2, lr2);

  float w1star = 200;
  float g1star = 0;
  float v1star = getInitialV(dp0, wd0, w1star);
  laggedPytorchUpdate(w1star, g1star, v1star, wd0, mm0, dp0, lr0);
  laggedPytorchUpdate(w1star, g1star, v1star, wd1, mm1, dp1, lr1);
  laggedPytorchUpdate(w1star, g1star, v1star, wd2, mm2, dp2, lr2);

  auto results  = getResults<float>(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
