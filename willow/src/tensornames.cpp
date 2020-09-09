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
  return id.substr(std::string(reservedGradientPrefix()).size());
}

bool isGradId(const TensorId &id) {
  const std::string pref = reservedGradientPrefix();
  const auto prefSize    = pref.size();
  return id.size() > prefSize && id.substr(0, prefSize) == pref;
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

std::vector<std::string> reservedOptimizerPrefixes() {
  return {// Optimizer
          reservedLossScalingPrefix(),
          // SGD0 / SGD1
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
          reservedSpecificAdamGradientScalingPrefix()};
}

std::vector<std::string> reservedPrefixes() {
  std::vector<std::string> prefs = {reservedGradientPrefix(),
                                    reservedUpdatedVarPrefix(),
                                    reservedStashedPrefix(),
                                    reservedRestoredPrefix(),
                                    reservedRandomSeedPrefix(),
                                    anchorSumPrefix(),
                                    cycleCountPrefix(),
                                    reservedRemoteArgPrefix()};

  std::vector<std::string> optPrefs;

  optPrefs = reservedOptimizerPrefixes();
  prefs.insert(prefs.end(), optPrefs.begin(), optPrefs.end());

  optPrefs = reservedOptimizerStatePrefixes();
  prefs.insert(prefs.end(), optPrefs.begin(), optPrefs.end());

  return prefs;
}

TensorId stripAllReservedPrefixes(TensorId id) {
  TensorId lastId    = id;
  TensorId currentId = id;
  do {
    lastId = currentId;
    for (auto prefix : reservedPrefixes()) {
      if (currentId.find(prefix) == 0) {
        currentId = currentId.substr(prefix.size());
      }
    }
  } while (lastId != currentId);
  return currentId;
}

std::vector<std::string> reservedOptimizerStatePrefixes() {
  std::vector<std::string> prefs = {reservedAcclPrefix(),
                                    reservedAccl1Prefix(),
                                    reservedAccl2Prefix(),
                                    reservedStepPrefix(),
                                    reservedAcclToReducePrefix(),
                                    reservedAcclToUpdatePrefix(),
                                    reservedAcclFinalOutPrefix()};
  return prefs;
}

std::vector<std::string> reservedAccumulatorPrefixes() {
  std::vector<std::string> prefs = {reservedAccumPrefix(),
                                    reservedAcclPrefix(),
                                    reservedAcclToReducePrefix(),
                                    reservedAcclToUpdatePrefix(),
                                    reservedAcclFinalOutPrefix()};
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
