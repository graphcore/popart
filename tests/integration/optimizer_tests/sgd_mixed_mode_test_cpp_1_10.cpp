// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE sgd_mixed_mode_test_1_10

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "get_results.hpp"
#include "popart/sgd.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(SgdMixedModeTestCpp1_10,
                              TestConfig,
                              SGD1And2TestConfigs) {
  // as test case 2, but with NO  :  gradient accumulation
  //                          YES :  graph replication

  float learningRate = 1.0f / 4.0f;
  popart::SGD opt0(
      {
          {"defaultLearningRate", {learningRate, false}},
          {"defaultMomentum", {1.0f, false}},
          {"defaultVelocityScaling", {14.15f, false}},
          {"lossScaling", {0.15f, false}},
      },
      {},
      TestConfig::sgdAccMm);
  auto opt1  = opt0;
  auto opt2  = opt0;
  bool wAccl = false;
  bool wRepl = true;

  auto results = getResults<float>(opt0, opt1, opt2, wAccl, wRepl);
  if (!acquisitionFailure(results)) {
    // int64_t correction = replicationFactor * accumulationFactor;
    int64_t correction =
        (wRepl ? replicationFactor : 1) * (wAccl ? accumulationFactor : 1);

    // including factor 2 for accumulation factor and 3 for replication factor
    auto absdiff0 =
        getAbsDiff(100 - correction * 6 * learningRate, std::get<0>(results));
    BOOST_CHECK(absdiff0 < 1e-5f);
    auto absdiff1 =
        getAbsDiff(200 - correction * 6 * learningRate, std::get<1>(results));
    BOOST_CHECK(absdiff1 < 1e-5f);
  } else {
    std::cout << "Failed to acquire device, test not run!";
  }
}
