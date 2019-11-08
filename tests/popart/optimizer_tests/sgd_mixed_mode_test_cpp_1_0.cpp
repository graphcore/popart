#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_0

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_0) {

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, true}},
      {"defaultMomentum", {1.0f, true}},
  });
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
