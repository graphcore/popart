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

bool isGradId(const TensorId &);

// inverse of previous function (non-grad name of grad tensor)
TensorId getNonGradId(const TensorId &tenId);

// get a recomputed tensor's name, based on original tensor
TensorId getRecompId(const TensorId &tenId);

// get an variable tensor's post-update name, based on original name
TensorId getUpdatedVarId(const TensorId &id);

constexpr const char *reservedGradientPrefix() { return "Gradient___"; }
constexpr const char *reservedUpdatedVarPrefix() { return "UpdatedVar___"; }
constexpr const char *reservedAccumPrefix() { return "Accum___"; }
constexpr const char *reservedAcclPrefix() { return "Accl___"; }
constexpr const char *reservedAccl1Prefix() { return "Accl1___"; }
constexpr const char *reservedAccl2Prefix() { return "Accl2___"; }
constexpr const char *reservedAccl3Prefix() { return "Accl3___"; }
constexpr const char *reservedStepPrefix() { return "Step___"; }
constexpr const char *reservedAcclToReducePrefix() {
  return "AcclIntoReduce___";
}
constexpr const char *reservedAcclToUpdatePrefix() {
  return "AcclIntoUpdate___";
}
constexpr const char *reservedAcclFinalOutPrefix() {
  return "AcclOutOfAcclUpdate___";
}
constexpr const char *reservedAdamUpdaterPrefix() { return "AdamUpdater___"; }
constexpr const char *reservedLambR1SqPrefix() { return "LambR1Sq___"; }
constexpr const char *reservedLambR2SqPrefix() { return "LambR2Sq___"; }

constexpr const char *reservedAdaptiveUpdaterPrefix() {
  return "AdaptiveUpdater___";
}

constexpr const char *reservedStashedPrefix() { return "Stashed___"; }
constexpr const char *reservedRestoredPrefix() { return "Restored___"; }

constexpr const char *reservedLossScalingPrefix() { return "lossScaling_"; }

constexpr const char *reservedRandomSeedPrefix() { return "randomSeed___"; }

constexpr const char *reservedSeedModifierPrefix() { return "SeedModifier___"; }

constexpr const char *reservedIndexPrefix() { return "Index___"; }

constexpr const char *reservedLoopCondPrefix() { return "LoopCond___"; }

constexpr const char *reservedLoopIteratorPrefix() { return "Iterator___"; }

constexpr const char *reservedInitPrefix() { return "Init___"; }

constexpr const char *reservedConcatInitPrefix() { return "ConcatInit___"; }

constexpr const char *reservedConstValuePrefix() { return "ConstValue___"; }

std::vector<std::string> reservedOptimizerPrefixes();
std::vector<std::string> reservedOptimizerStatePrefixes();
std::vector<std::string> reservedAccumulatorPrefixes();
std::vector<std::string> reservedPrefixes();

TensorId stripAllReservedPrefixes(TensorId id);

TensorId getRemoteArgTensorId(TensorId base_id);
constexpr const char *reservedRemoteArgPrefix() { return "RemoteArg___"; }

TensorId getNonRemoteArgTensorId(const TensorId &id);

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

constexpr const char *reservedDefaultLearningRatePrefix() {
  return "learningRate___default___";
}
constexpr const char *reservedSpecificLearningRatePrefix() {
  return "learningRate___specific___";
}

constexpr const char *reservedDefaultWeightDecayPrefix() {
  return "weightDecay___default___";
}
constexpr const char *reservedSpecificWeightDecayPrefix() {
  return "weightDecay___specific___";
}

constexpr const char *reservedDefaultLossScalingPrefix() {
  return "lossScaling___default___";
}
constexpr const char *reservedSpecificLossScalingPrefix() {
  return "lossScaling___specific___";
}

constexpr const char *reservedDefaultMaxWeightNormPrefix() {
  return "maxWeightNorm___default___";
}
constexpr const char *reservedSpecificMaxWeightNormPrefix() {
  return "maxWeightNorm___specific___";
}

constexpr const char *reservedDefaultAdamBeta1Prefix() {
  return "adamBeta1___default___";
}
constexpr const char *reservedSpecificAdamBeta1Prefix() {
  return "adamBeta1___specific___";
}

constexpr const char *reservedDefaultAdamBeta2Prefix() {
  return "adamBeta2___default___";
}
constexpr const char *reservedSpecificAdamBeta2Prefix() {
  return "adamBeta2___specific___";
}

constexpr const char *reservedDefaultAdamEpsPrefix() {
  return "adamEps___default___";
}
constexpr const char *reservedSpecificAdamEpsPrefix() {
  return "adamEps___specific___";
}

constexpr const char *reservedDefaultAdamGradientScalingPrefix() {
  return "adamGradientScaling___default___";
}
constexpr const char *reservedSpecificAdamGradientScalingPrefix() {
  return "adamGradientScaling___specific___";
}

constexpr const char *reservedDefaultAdaptiveAlphaPrefix() {
  return "adaptiveAlpha___default___";
}
constexpr const char *reservedSpecificAdaptiveAlphaPrefix() {
  return "adaptiveAlpha___specific___";
}

constexpr const char *reservedDefaultAdaptiveMomentumPrefix() {
  return "adaptiveMomentum___default___";
}
constexpr const char *reservedSpecificAdaptiveMomentumPrefix() {
  return "adaptiveMomentum___specific___";
}

constexpr const char *reservedDefaultAdaptiveEpsPrefix() {
  return "adaptiveEps___default___";
}
constexpr const char *reservedSpecificAdaptiveEpsPrefix() {
  return "adaptiveEps___specific___";
}

constexpr const char *reservedDefaultAdaptiveGradientScalingPrefix() {
  return "adaptiveGradientScaling___default___";
}
constexpr const char *reservedSpecificAdaptiveGradientScalingPrefix() {
  return "adaptiveGradientScaling___specific___";
}

constexpr const char *reservedDefaultStepPrefix() {
  return "step___default___";
}
constexpr const char *reservedSpecificStepPrefix() {
  return "step___specific___";
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
