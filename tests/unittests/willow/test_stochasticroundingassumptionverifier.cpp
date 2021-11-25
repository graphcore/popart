// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Unittest_StochasticRoundingAssumptionVerifier

#include <string>

#include <boost/test/unit_test.hpp>

#include <popart/error.hpp>
#include <popart/ir.hpp>

#include <stochasticroundingassumptionverifier.hpp>

#include <testutil/test_graphs/graph_test_models.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(verify) {

  // Test that `StochasticRoundingAssumptionVerifier::verify` throws an
  // exception when either `stochasticRoundingEnabled` is false but an Op has
  // `stochasticRoundingMethod` set or`stochasticRoundingEnabled` is true but an
  // Op does not have `stochasticRoundingMethod` set.

  enum class StochasticRounding { Yes, No };

  enum class StochasticRoundingMethodSet { All, Some, None };

  enum class ExpectError { Yes, No };

  auto test = [](StochasticRounding sr,
                 StochasticRoundingMethodSet set,
                 ExpectError err) {
    GraphTestModel5 model(GraphTestModel5::SG1::No, GraphTestModel5::SG2::No);
    auto &ir = model.getIr();

    // Enable stochastic rounding in IR depending on parameter 'sr'
    SessionOptions opts;
    opts.enableStochasticRounding = (sr == StochasticRounding::Yes);
    ir.setUserOptions(opts);

    // Set stochastic rounding method attribute in accordance with 'set'.
    auto ops = ir.getAllOps();
    if (set == StochasticRoundingMethodSet::All) {
      ops[0]->setStochasticRoundingMethod(
          StochasticRoundingMethod::DifferingSeeds);
    }
    if (set != StochasticRoundingMethodSet::None) {
      for (size_t i = 1; i < ops.size(); ++i) {
        ops.at(i)->setStochasticRoundingMethod(
            StochasticRoundingMethod::DifferingSeeds);
      }
    }

    // Test for presence/absence of exception.
    StochasticRoundingAssumptionVerifier verifier{ir};
    if (err == ExpectError::Yes) {
      BOOST_REQUIRE_THROW(verifier.verify(), popart::error);
    } else {
      verifier.verify();
    }
  };

  test(StochasticRounding::Yes,
       StochasticRoundingMethodSet::None,
       ExpectError::Yes);
  test(StochasticRounding::Yes,
       StochasticRoundingMethodSet::Some,
       ExpectError::Yes);
  test(StochasticRounding::Yes,
       StochasticRoundingMethodSet::All,
       ExpectError::No);

  test(StochasticRounding::No,
       StochasticRoundingMethodSet::None,
       ExpectError::No);
  test(StochasticRounding::No,
       StochasticRoundingMethodSet::Some,
       ExpectError::Yes);
  test(StochasticRounding::No,
       StochasticRoundingMethodSet::All,
       ExpectError::Yes);
}
