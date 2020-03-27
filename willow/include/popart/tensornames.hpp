// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORNAMES_HPP
#define GUARD_NEURALNET_TENSORNAMES_HPP

#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

// where tensor tenId is consumed by Op with OpId opId at
// index "index", what should the name of the edge-gradient
// along this edge be? This is pure string manipulation.
TensorId getEdgeGradId(TensorId tenId, OpId opId, int index);

// the name of the tensor which is the
// total gradient of a forward tensor
TensorId getGradId(const TensorId &tenId);

// inverse of previous function (non-grad name of grad tensor)
TensorId getNonGradId(const TensorId &tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(const TensorId &tenId);

// get an variable tensor's post-update name, based on original name
TensorId getUpdatedVarId(const TensorId &id);

constexpr const char *reservedGradientPrefix() { return "Gradient___"; }
constexpr const char *reservedUpdatedVarPrefix() { return "UpdatedVar___"; }
constexpr const char *reservedAcclToAccumulatorPrefix() {
  return "AcclIntoAccumulator___";
}
constexpr const char *reservedAcclToReducePrefix() {
  return "AcclIntoReduce___";
}
constexpr const char *reservedAcclToUpdatePrefix() {
  return "AcclIntoUpdate___";
}
constexpr const char *reservedAcclFinalOutPrefix() {
  return "AcclOutOfAcclUpdate___";
}
constexpr const char *reservedStashedPrefix() { return "Stashed___"; }
constexpr const char *reservedRestoredPrefix() { return "Restored___"; }

constexpr const char *reservedLossScalingPrefix() { return "lossScaling_"; }

constexpr const char *reservedRandomSeedPrefix() { return "randomSeed___"; }

std::vector<std::string> reservedOptimizerPrefixes();
std::vector<std::string> reservedPrefixes();

TensorId getCacheArgTensorId(TensorId base_id);

TensorId createRecomputedTensorId(TensorId base_id);

constexpr const char *reservedDefaultWeightDecayScaleFactor0Prefix() {
  return "weightDecayScaleFactor0___default___";
}
constexpr const char *reservedSpecificWeightDecayScaleFactor0Prefix() {
  return "weightDecayScaleFactor0___specific___";
}

constexpr const char *reservedDefaultScaledLearningRate0Prefix() {
  return "scaledLearningRate0___default___";
}
constexpr const char *reservedSpecificScaledLearningRate0Prefix() {
  return "scaledLearningRate0___specific___";
}

constexpr const char *reservedDefaultScaledWeightDecay1Prefix() {
  return "scaledWeightDecay1___default___";
}
constexpr const char *reservedSpecificScaledWeightDecay1Prefix() {
  return "scaledWeightDecay1___specific___";
}

constexpr const char *reservedDefaultScaledLearningRate1Prefix() {
  return "scaledLearningRate1___default___";
}
constexpr const char *reservedSpecificScaledLearningRate1Prefix() {
  return "scaledLearningRate1___specific___";
}

constexpr const char *reservedDefaultDampeningScaleFactor1Prefix() {
  return "dampeningScaleFactor1___default___";
}
constexpr const char *reservedSpecificDampeningScaleFactor1Prefix() {
  return "dampeningScaleFactor1___specific___";
}

constexpr const char *reservedDefaultScaledMomentum1Prefix() {
  return "scaledMomentum1___default___";
}
constexpr const char *reservedSpecificScaledMomentum1Prefix() {
  return "scaledMomentum1___specific___";
}

constexpr const char *hostReduceGradCopyPrefix() {
  return "hostReduceGradCopy___";
}
constexpr const char *hostReduceVarCopyPrefix() {
  return "hostReduceVarCopy___";
}

constexpr const char *anchorSumPrefix() { return "anchorSum___"; }

constexpr const char *cycleCountPrefix() { return "cycleCount___"; }

} // namespace popart

#endif
