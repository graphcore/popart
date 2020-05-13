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
  return {reservedDefaultScaledLearningRate0Prefix(),
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

          reservedLossScalingPrefix()};
}

std::vector<std::string> reservedPrefixes() {
  std::vector<std::string> prefs = {reservedGradientPrefix(),
                                    reservedUpdatedVarPrefix(),
                                    reservedAcclToAccumulatorPrefix(),
                                    reservedAcclToReducePrefix(),
                                    reservedAcclToUpdatePrefix(),
                                    reservedAcclFinalOutPrefix(),
                                    reservedStashedPrefix(),
                                    reservedRestoredPrefix(),
                                    reservedRandomSeedPrefix(),
                                    anchorSumPrefix(),
                                    cycleCountPrefix(),
                                    reservedCacheArgPrefix()};

  std::vector<std::string> optPrefs = reservedOptimizerPrefixes();
  prefs.insert(prefs.end(), optPrefs.begin(), optPrefs.end());

  return prefs;
}

TensorId getCacheArgTensorId(TensorId base_id) {
  auto ca_id = logging::format("{}{}", reservedCacheArgPrefix(), base_id);
  logging::ir::trace("Generating tensor id {}", ca_id);
  return ca_id;
}

TensorId getNonCacheArgTensorId(const TensorId &id) {
  return id.substr(std::string(reservedCacheArgPrefix()).size());
}

TensorId createRecomputedTensorId(TensorId base_id) {
  auto recompute_id = logging::format("{}__re", base_id);
  logging::ir::trace("Generating tensor id {}", recompute_id);
  return recompute_id;
}

} // namespace popart
