// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_SGD0_CompoundScalarHelpers
#include <boost/test/unit_test.hpp>
#include <popart/compoundscalarhelper.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/sgd.hpp>

using namespace popart;

namespace {
void validate(SGD sgd, float expected_lr, float expected_wd) {
  ScaledLearningRate0Helper lr;
  WeightDecayScaleFactor0Helper wd;

  BOOST_CHECK_EQUAL(lr.val("", sgd), expected_lr);
  BOOST_CHECK_EQUAL(wd.val("", sgd), expected_wd);
}
} // namespace

BOOST_AUTO_TEST_CASE(TestSGD0_Base) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_lossScaling) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}},
           {"lossScaling", {4.0f, true}}}};
  SessionOptions opts;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f / 4.0f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_ReplicaSum) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableReplicatedGraphs = true;
  opts.replicatedGraphCount   = 2;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_ReplicaMeanPost) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f / 2.0f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_ReplicaMeanRunning) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableReplicatedGraphs                  = true;
  opts.replicatedGraphCount                    = 2;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_AccumSum) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableReplicatedGraphs = true;
  opts.replicatedGraphCount   = 2;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_AccumMeanPost) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Post;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f / 4.0f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_AccumMeanRunning) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}}}};
  SessionOptions opts;
  opts.enableGradientAccumulation              = true;
  opts.accumulationFactor                      = 4;
  opts.accumulationAndReplicationReductionType = ReductionType::Mean;
  opts.meanAccumulationAndReplicationReductionStrategy =
      MeanReductionStrategy::Running;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, 0.1f, 1.0f - (0.1f * 0.01f));
}

BOOST_AUTO_TEST_CASE(TestSGD0_Dampening) {
  SGD sgd{{{"defaultLearningRate", {0.1f, true}},
           {"defaultWeightDecay", {0.01f, true}},
           {"defaultDampening", {0.2f, true}},
           {"defaultMomentum", {0.0f, true}}}};
  SessionOptions opts;
  sgd.setFactorsFromOptions(opts);
  validate(sgd, (1.0f - 0.2f) * 0.1f, 1.0f - ((1.0f - 0.2f) * 0.1f * 0.01f));
}