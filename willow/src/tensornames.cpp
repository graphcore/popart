// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <popart/logging.hpp>
#include <popart/tensornames.hpp>

namespace popart {

TensorId getGradId(const TensorId &id) { return reservedGradientPrefix() + id; }

TensorId getUpdatedVarId(const TensorId &id) {
  return reservedUpdatedVarPrefix() + id;
}

TensorId getNonGradId(const TensorId &id) {
  // TODO : constexpr the size of this string T8265
  return id.str().substr(std::string(reservedGradientPrefix()).size());
}

bool isGradId(const TensorId &id) {
  const std::string pref = reservedGradientPrefix();
  const auto prefSize    = pref.size();
  return id.str().size() > prefSize && id.str().substr(0, prefSize) == pref;
}

TensorId getEdgeGradId(TensorId tenId, OpId opId, int index) {
  // we don't need the name of the tensor which this is an edge-grad of,
  // the edge-gradient is uniquely defined by the the edge it flows on
  // in the forward pass (input at 'index' to 'opId')
  (void)tenId;
  std::stringstream ss;
  ss << reservedGradientPrefix() << opId << '_' << index;
  TensorId edgeGradId = ss.str();
  return edgeGradId;
}

const std::vector<std::string> &reservedOptimizerPrefixes() {
  const static std::vector<std::string> result = {
      // Optimizer
      reservedLossScalingPrefix(),
      reservedAutomaticLossScalePrefix(),
      // SGD0 / SGD1 / SGD2
      reservedDefaultScaledLearningRate0Prefix(),
      reservedSpecificScaledLearningRate0Prefix(),
      reservedDefaultWeightDecayScaleFactor0Prefix(),
      reservedSpecificWeightDecayScaleFactor0Prefix(),
      reservedDefaultScaledLearningRate1Prefix(),
      reservedSpecificScaledLearningRate1Prefix(),
      reservedDefaultScaledWeightDecay1Prefix(),
      reservedSpecificScaledWeightDecay1Prefix(),
      reservedDefaultDampeningScaleFactor1Prefix(),
      reservedSpecificDampeningScaleFactor1Prefix(),
      reservedDefaultScaledMomentum1Prefix(),
      reservedSpecificScaledMomentum1Prefix(),
      reservedDefaultScaledLearningRate2Prefix(),
      reservedSpecificScaledLearningRate2Prefix(),
      reservedDefaultDampeningScaleFactor2Prefix(),
      reservedSpecificDampeningScaleFactor2Prefix(),
      reservedDefaultScaledMomentum2Prefix(),
      reservedSpecificScaledMomentum2Prefix(),
      // Adam / Lamb
      reservedDefaultAdamBeta1Prefix(),
      reservedSpecificAdamBeta1Prefix(),
      reservedDefaultAdamBeta2Prefix(),
      reservedSpecificAdamBeta2Prefix(),
      reservedDefaultLearningRatePrefix(),
      reservedSpecificLearningRatePrefix(),
      reservedDefaultAdamEpsPrefix(),
      reservedSpecificAdamEpsPrefix(),
      reservedDefaultWeightDecayPrefix(),
      reservedSpecificWeightDecayPrefix(),
      reservedDefaultAdamGradientScalingPrefix(),
      reservedSpecificAdamGradientScalingPrefix(),
      reservedDefaultMaxWeightNormPrefix(),
      reservedSpecificMaxWeightNormPrefix(),
      // Adaptive
      reservedDefaultAdaptiveAlphaPrefix(),
      reservedSpecificAdaptiveAlphaPrefix(),
      reservedDefaultAdaptiveEpsPrefix(),
      reservedSpecificAdaptiveEpsPrefix(),
      reservedDefaultAdaptiveGradientScalingPrefix(),
      reservedSpecificAdaptiveGradientScalingPrefix(),
      reservedDefaultAdaptiveMomentumPrefix(),
      reservedSpecificAdaptiveMomentumPrefix()};
  return result;
}

const std::vector<std::string> &reservedPrefixes() {
  const static std::vector<std::string> prefs = []() {
    std::vector<std::string> prefs = {reservedGradientPrefix(),
                                      reservedUpdatedVarPrefix(),
                                      reservedStashedPrefix(),
                                      reservedRestoredPrefix(),
                                      reservedRandomSeedPrefix(),
                                      anchorSumPrefix(),
                                      cycleCountPrefix(),
                                      reservedRemoteArgPrefix()};

    const auto &optPrefs = reservedOptimizerPrefixes();
    prefs.insert(prefs.end(), optPrefs.begin(), optPrefs.end());

    const auto &optStatePrefs = reservedOptimizerStatePrefixes();
    prefs.insert(prefs.end(), optStatePrefs.begin(), optStatePrefs.end());
    return prefs;
  }();

  return prefs;
}

TensorId stripAllReservedPrefixes(TensorId id) {
  TensorId lastId    = id;
  TensorId currentId = id;
  do {
    lastId = currentId;
    for (auto prefix : reservedPrefixes()) {
      if (currentId.str().find(prefix) == 0) {
        currentId = currentId.str().substr(prefix.size());
      }
    }
  } while (lastId != currentId);
  return currentId;
}

const std::vector<std::string> &reservedOptimizerStatePrefixes() {
  const static std::vector<std::string> prefs = {reservedAcclPrefix(),
                                                 reservedAccl1Prefix(),
                                                 reservedAccl2Prefix(),
                                                 reservedAccl3Prefix(),
                                                 reservedStepPrefix(),
                                                 reservedAcclToReducePrefix(),
                                                 reservedAcclToUpdatePrefix(),
                                                 reservedAcclFinalOutPrefix()};
  return prefs;
}

const std::vector<std::string> &reservedAccumulatorPrefixes() {
  const static std::vector<std::string> prefs = {
      reservedAccumPrefix(),
      // Accumulator/momentum combination tensors,
      // specifically used for the SGD1 optimizer
      reservedAcclPrefix(),
      reservedAcclToReducePrefix(),
      reservedAcclToUpdatePrefix(),
      reservedAcclFinalOutPrefix(),
      reservedCounterPrefix()};
  return prefs;
}

TensorId getRemoteArgTensorId(TensorId base_id) {
  auto ca_id = logging::format("{}{}", reservedRemoteArgPrefix(), base_id);
  logging::ir::trace("Generating tensor id {}", ca_id);
  return ca_id;
}

TensorId createRecomputedTensorId(TensorId base_id) {
  auto recompute_id = logging::format("{}__re", base_id);
  logging::ir::trace("Generating tensor id {}", recompute_id);
  return recompute_id;
}

} // namespace popart
