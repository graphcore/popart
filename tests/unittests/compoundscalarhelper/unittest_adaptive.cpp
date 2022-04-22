// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Adaptive_CompoundScalarHelpers
#include <boost/test/unit_test.hpp>
#include <string>
#include <utility>
#include <popart/adaptive.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/sessionoptions.hpp>

#include "popart/datatype.hpp"
#include "popart/op.hpp"
#include "popart/optimizer.hpp"
#include "popart/optimizervalue.hpp"

using namespace popart;

namespace {
void validate(Adaptive adaptive, float expected_lr, float expected_gs) {
  AdaptiveLearningRateHelper lr;
  AdaptiveGradientScalingHelper gs;

  BOOST_CHECK_EQUAL(lr.val("", adaptive), expected_lr);
  BOOST_CHECK_EQUAL(gs.val("", adaptive), expected_gs);
}
} // namespace

BOOST_AUTO_TEST_CASE(TestAdaptive_Base) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_lossScaling) {
  Adaptive adaptive{
      {{"defaultLearningRate", {0.1f, true}}, {"lossScaling", {4.0f, true}}},
      AdaptiveMode::AdaGrad,
      WeightDecayMode::Decay,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT,
      DataType::FLOAT};
  SessionOptions opts;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f / 4.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_ReplicaSum) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs = true;
  opts.replicatedGraphCount   = 2;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_ReplicaMeanPost) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f / 2.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_ReplicaMeanRunning) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_AccumSum) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation = true;
  opts.accumulationFactor         = 4;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_AccumMeanPost) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f / 4.0f);
}

BOOST_AUTO_TEST_CASE(TestAdaptive_AccumMeanRunning) {
  Adaptive adaptive{{
                        {"defaultLearningRate", {0.1f, true}},
                    },
                    AdaptiveMode::AdaGrad,
                    WeightDecayMode::Decay,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT,
                    DataType::FLOAT};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  adaptive.setFactorsFromOptions(opts);
  validate(adaptive, 0.1f, 1.0f);
}
