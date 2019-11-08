#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_9

#include "get_results.hpp"

BOOST_AUTO_TEST_CASE(SgdMixedModeTestCpp1_9) {

  // As in test 6, but with graph replication and gradient accumulation

  // const weight decay, different on the 2 weights
  float defaultWd = 0.02f;
  float weight1Wd = 0.05f;
  // variable default momentum, different on the 2 weights
  float defaultMm0 = 0.9f;
  float defaultMm1 = 0.8f;
  float defaultMm2 = 0.7f;
  float weight0Mm  = 0.6f; // constant momentum for weight 0
  // constant dampening, the same on the 2 weights
  float dp = 0.05f;
  // variable learning rate, different on the 2 weights
  float defaultLr0 = 1.0;
  float defaultLr1 = 0.64;
  float defaultLr2 = 0.8;
  float weight0Lr0 = 0.7;
  float weight0Lr1 = 0.64;
  float weight0Lr2 = 0.32;
  // constant loss scaling
  float ls = 0.2f;
  // constant velocity scaling, different on the 2 weights
  float defaultVs = 0.25;
  float weight0Vs = 0.35;

  popart::SGD opt0({{"defaultDampening", {dp, true}},
                    {"defaultLearningRate", {defaultLr0, false}},
                    {"defaultWeightDecay", {defaultWd, true}},
                    {"defaultVelocityScaling", {defaultVs, true}},
                    {"lossScaling", {ls, true}},
                    {"defaultMomentum", {defaultMm0, false}}});

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
                    {"defaultMomentum", {defaultMm1, false}}});
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
                    {"defaultMomentum", {defaultMm2, false}}});
  opt2.insertSpecific(w0name,
                      {{"momentum", {weight0Mm, true}},
                       {"learningRate", {weight0Lr2, false}},
                       {"velocityScaling", {weight0Vs, true}}});
  opt2.insertSpecific(w1name, {{"weightDecay", {weight1Wd, true}}});

  auto getInitialV = [](float dp, float wd, float W) {
    // no need to include vs, it is constant so we can use vanilla pytorch
    // update equations
    return (1.0f - dp) * wd * W;
  };

  float w0star = 100;
  float g0star = 0;
  float v0star = getInitialV(dp, defaultWd, w0star);
  laggedPytorchUpdate(w0star,
                      g0star,
                      v0star,
                      defaultWd,
                      weight0Mm,
                      dp,
                      weight0Lr0,
                      replicationFactor,
                      accumulationFactor);

  laggedPytorchUpdate(w0star,
                      g0star,
                      v0star,
                      defaultWd,
                      weight0Mm,
                      dp,
                      weight0Lr1,
                      replicationFactor,
                      accumulationFactor);

  laggedPytorchUpdate(w0star,
                      g0star,
                      v0star,
                      defaultWd,
                      weight0Mm,
                      dp,
                      weight0Lr2,
                      replicationFactor,
                      accumulationFactor);

  float w1star = 200;
  float g1star = 0;
  float v1star = getInitialV(dp, weight1Wd, w1star);
  laggedPytorchUpdate(w1star,
                      g1star,
                      v1star,
                      weight1Wd,
                      defaultMm0,
                      dp,
                      defaultLr0,
                      replicationFactor,
                      accumulationFactor);

  laggedPytorchUpdate(w1star,
                      g1star,
                      v1star,
                      weight1Wd,
                      defaultMm1,
                      dp,
                      defaultLr1,
                      replicationFactor,
                      accumulationFactor);

  laggedPytorchUpdate(w1star,
                      g1star,
                      v1star,
                      weight1Wd,
                      defaultMm2,
                      dp,
                      defaultLr2,
                      replicationFactor,
                      accumulationFactor);

  // test with float32
  auto results = getResults<popart::float16_t>(opt0, opt1, opt2, true, true);
  if (results != acquisitionFailure) {
    auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
    auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
    std::cout << "abs diffs at float16: " << absdiff0 << " and " << absdiff1
              << std::endl;
    BOOST_CHECK(absdiff0 < 1e-1f);
    BOOST_CHECK(absdiff1 < 1e-1f);
  } else {
    std::cout << "Failed to acquire device, test not run!";
  }

  // test with float32
  results = getResults<float>(opt0, opt1, opt2, true, true);
  if (results != acquisitionFailure) {
    auto absdiff0 = getAbsDiff(w0star, std::get<0>(results));
    auto absdiff1 = getAbsDiff(w1star, std::get<1>(results));
    std::cout << "abs diffs at float32: " << absdiff0 << " and " << absdiff1
              << std::endl;
    BOOST_CHECK(absdiff0 < 1e-4f);
    BOOST_CHECK(absdiff1 < 1e-4f);
  } else {
    std::cout << "Failed to acquire device, test not run!";
  }
}
