#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_1

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_1) {
  // as test case 0, but with loss scaling and velocity scaling

  float lrTest0 = 1.0f / 4.0f;
  popart::SGD opt0({
      {"defaultLearningRate", {lrTest0, true}},
      {"defaultMomentum", {1.0f, true}},
      {"defaultVelocityScaling", {14.15f, true}},
      {"lossScaling", {0.15f, true}},
  });
  auto opt1     = opt0;
  auto opt2     = opt0;
  auto results  = getResults(opt0, opt1, opt2, false, false);
  auto absdiff0 = getAbsDiff(100 - 6 * lrTest0, std::get<0>(results));
  BOOST_CHECK(absdiff0 < 1e-5f);
  auto absdiff1 = getAbsDiff(200 - 6 * lrTest0, std::get<1>(results));
  BOOST_CHECK(absdiff1 < 1e-5f);
}
