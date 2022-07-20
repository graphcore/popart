// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <popart/logging.hpp>
#include <popart/tensornames.hpp>

#include "popart/names.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/util.hpp"

namespace popart {
class Graph;

TensorId fwdIdToBwdGradId(const Graph &fwdGraph,
                          const Graph &bwdGraph,
                          const TensorId &fwdId) {
  auto x = removeScope(fwdGraph, fwdId);
  x      = getGradId(x);
  return addScope(bwdGraph, x);
}

TensorId bwdGradIdToFwdId(const Graph &fwdGraph,
                          const Graph &bwdGraph,
                          const TensorId &bwdId) {
  auto x = removeScope(bwdGraph, bwdId);
  x      = getNonGradId(x);
  return addScope(fwdGraph, x);
}

TensorId fwdIdToClonedBwdId(const Graph &fwdGraph,
                            const Graph &bwdGraph,
                            const TensorId &fwdId) {
  return addScope(bwdGraph, removeScope(fwdGraph, fwdId));
}

TensorId bwdNonGradIdToFwdId(const Graph &fwdGraph,
                             const Graph &bwdGraph,
                             const TensorId &bwdId) {
  auto x = removeScope(bwdGraph, bwdId);
  return addScope(fwdGraph, x);
}

TensorId getGradId(const TensorId &id) { return reservedGradientPrefix() + id; }

TensorId getNonGradId(const TensorId &id) {
  // TODO : constexpr the size of this string T8265
  return id.substr(std::string(reservedGradientPrefix()).size());
}

bool isGradId(const TensorId &id) {
  const std::string pref = reservedGradientPrefix();
  const auto prefSize    = pref.size();
  return id.size() > prefSize && id.substr(0, prefSize) == pref;
}

TensorId getEdgeGradId(const OpId &opId, const int &index) {
  std::stringstream ss;
  ss << reservedGradientPrefix() << opId << '_' << index;
  TensorId edgeGradId = ss.str();
  return edgeGradId;
}

// Functions using reserved prefixes

TensorId getRemoteArgTensorId(const TensorId &base_id) {
  auto ca_id = logging::format("{}{}", reservedRemoteArgPrefix(), base_id);
  logging::ir::trace("Generating tensor id {}", ca_id);
  return ca_id;
}

TensorId getUpdatedVarId(const TensorId &id) {
  return reservedUpdatedVarPrefix() + id;
}

// Functions creating suffixes

TensorId createRecomputedTensorId(const TensorId &base_id) {
  auto recompute_id = logging::format("{}__re", base_id);
  logging::ir::trace("Generating tensor id {}", recompute_id);
  return recompute_id;
}

// Removing of prefixes

TensorId stripAllReservedPrefixes(const TensorId &id) {
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

// Collection of prefixes

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
      reservedDefaultSGDWeightDecayPrefix(),
      reservedSpecificSGDWeightDecayPrefix(),
      reservedDefaultSGDMomentumPrefix(),
      reservedSpecificSGDMomentumPrefix(),
      reservedDefaultNesterovGradScaleFactor1Prefix(),
      reservedSpecificNesterovGradScaleFactor1Prefix(),
      reservedDefaultNesterovGradScaleFactor2Prefix(),
      reservedSpecificNesterovGradScaleFactor2Prefix(),
      reservedDefaultNesterovDampeningScaleFactor1Prefix(),
      reservedSpecificNesterovDampeningScaleFactor1Prefix(),
      reservedDefaultNesterovDampeningScaleFactor2Prefix(),
      reservedSpecificNesterovDampeningScaleFactor2Prefix(),
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

const std::vector<std::string> &uncategorizedReservedPrefixes() {
  const static std::vector<std::string> prefs = {
      reservedAccl1Prefix(),
      reservedAccl2Prefix(),
      reservedAccl3Prefix(),
      reservedAdamUpdaterPrefix(),
      reservedAdaptiveUpdaterPrefix(),
      anchorSumPrefix(),
      anchorFinalPrefix(),
      reservedConcatInitPrefix(),
      reservedConstValuePrefix(),
      cycleCountPrefix(),
      reservedFinalReducedGradPrefix(),
      reservedGlobalNormPrefix(),
      reservedGradientPrefix(),
      reservedIndexPrefix(),
      reservedInitPrefix(),
      reservedLoopIteratorPrefix(),
      reservedLambR1SqPrefix(),
      reservedLambR2SqPrefix(),
      reservedLoopCondPrefix(),
      reservedDefaultLossScalingPrefix(),
      reservedSpecificLossScalingPrefix(),
      reservedLossScalingRatioPrefix(),
      reservedPreviousLossScalingPrefix(),
      reservedRandomSeedPrefix(),
      reservedSeedModifierPrefix(),
      reservedRemoteArgPrefix(),
      reservedRestoredPrefix(),
      reservedStashedPrefix(),
      reservedStepPrefix(),
      reservedDefaultStepPrefix(),
      reservedSpecificStepPrefix(),
      reservedUpdatedVarPrefix()};
  return prefs;
}

const std::vector<std::string> &reservedPrefixes() {
  const static std::vector<std::string> prefs = []() {
    std::vector<std::string> prefs = uncategorizedReservedPrefixes();

    const auto &optPrefs = reservedOptimizerPrefixes();
    prefs.insert(prefs.end(), optPrefs.begin(), optPrefs.end());

    const auto &optStatePrefs = reservedOptimizerStatePrefixes();
    prefs.insert(prefs.end(), optStatePrefs.begin(), optStatePrefs.end());

    const auto &accumulatorPrefs = reservedAccumulatorPrefixes();
    prefs.insert(prefs.end(), accumulatorPrefs.begin(), accumulatorPrefs.end());

    return prefs;
  }();

  return prefs;
}

} // namespace popart
