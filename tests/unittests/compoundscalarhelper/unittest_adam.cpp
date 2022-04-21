// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Adam_CompoundScalarHelpers
#include <boost/test/unit_test.hpp>
#include <popart/adam.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/sessionoptions.hpp>

using namespace popart;

namespace {
void validate(const Adam &adam,
              float expected_lr,
              float expected_gs,
              float expected_eps) {
  AdamLearningRateHelper lr;
  AdamGradientScalingHelper gs;
  AdamEpsHelper eps;

  BOOST_CHECK_EQUAL(lr.val("", adam), expected_lr);
  BOOST_CHECK_EQUAL(gs.val("", adam), expected_gs);
  BOOST_CHECK_EQUAL(eps.val("", adam), expected_eps);
}
} // namespace

BOOST_AUTO_TEST_CASE(TestAdam_Base) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_lossScaling) {
  Adam adam{{{"defaultLearningRate", {0.1f, true}},
             {"defaultEps", {1e-3f, true}},
             {"lossScaling", {4.0f, true}}},
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f / 4.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_ReplicaSum) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs = true;
  opts.replicatedGraphCount   = 2;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_ReplicaMeanPost) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f / 2.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_ReplicaMeanRunning) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_AccumSum) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 4;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_AccumMeanPost) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f / 4.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_AccumMeanRunning) {
  Adam adam{{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultEps", {1e-3f, true}},
            },
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f);
}

BOOST_AUTO_TEST_CASE(TestAdam_scaledOptimizerState) {
  Adam adam{std::map<std::string, std::pair<float, bool>>{
                {"defaultLearningRate", {0.1f, true}},
                {"defaultWeightDecay", {0.01f, true}},
                {"defaultEps", {1e-3f, true}},
                {"lossScaling", {4.0f, true}}},
            AdamMode::Adam,
            WeightDecayMode::Decay,
            DataType::FLOAT,
            DataType::FLOAT,
            DataType::FLOAT,
            {},
            true};
  SessionOptions opts;
  adam.setFactorsFromOptions(opts);
  validate(adam, 0.1f, 1.0f, 1e-3f * 4.0f);
}
