// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>

#include <boost/lexical_cast.hpp>

#include <popart/error.hpp>
#include <popart/sessionoptions.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

AutomaticLossScalingSettings::AutomaticLossScalingSettings(
    bool enabled_,
    const nonstd::optional<std::vector<TensorId>> &toTrackTensors_,
    float binEdgeLocation_,
    float thresholdUpperCountProportion_)
    : enabled{enabled_}, binEdgeLocation{binEdgeLocation_},
      thresholdUpperCountProportion{thresholdUpperCountProportion_},
      toTrackTensors{toTrackTensors_} {}

std::size_t AutomaticLossScalingSettings::hash() const {
  std::size_t seed = 0;
  boost::hash_combine(seed, enabled);
  boost::hash_combine(seed, binEdgeLocation);
  boost::hash_combine(seed, thresholdUpperCountProportion);
  if (toTrackTensors) {
    boost::hash_combine(seed, toTrackTensors.value());
  }
  return seed;
}

TensorLocationSettings::TensorLocationSettings(
    TensorLocation location_,
    int minElementsForOffChip_,
    int minElementsForReplicatedTensorSharding_)
    : location{location_}, minElementsForOffChip{minElementsForOffChip_},
      minElementsForReplicatedTensorSharding{
          minElementsForReplicatedTensorSharding_} {}

TensorLocationSettings::TensorLocationSettings(
    TensorStorage storage_,
    int minElementsForOffChip_,
    int minElementsForReplicatedTensorSharding_)
    : location{storage_}, minElementsForOffChip{minElementsForOffChip_},
      minElementsForReplicatedTensorSharding{
          minElementsForReplicatedTensorSharding_} {}

BatchSerializationSettings::BatchSerializationSettings(
    int factor_,
    bool concatOnVirtualGraphChange_,
    bool concatOnExecutionPhaseChange_,
    bool concatOnPipelineStageChange_,
    BatchSerializationTransformContext transformContext_,
    BatchSerializationMethod method_,
    BatchSerializationBatchSchedule batchSchedule_)
    : factor{factor_}, concatOnVirtualGraphChange{concatOnVirtualGraphChange_},
      concatOnExecutionPhaseChange{concatOnExecutionPhaseChange_},
      concatOnPipelineStageChange{concatOnPipelineStageChange_},
      transformContext{transformContext_}, method{method_},
      batchSchedule{batchSchedule_} {}

std::string toString(VirtualGraphMode v) {
  switch (v) {
  case VirtualGraphMode::Off:
    return "VirtualGraphMode::Off";
  case VirtualGraphMode::Manual:
    return "VirtualGraphMode::Manual";
  case VirtualGraphMode::Auto:
    return "VirtualGraphMode::Auto";
  case VirtualGraphMode::ExecutionPhases:
    return "VirtualGraphMode::ExecutionPhases";
  case VirtualGraphMode::N:
    throw error("Bad VirtualGraphMode {}", static_cast<int>(v));
  default:
    throw error("Unknown VirtualGraphMode");
  }
}

std::ostream &operator<<(std::ostream &os, VirtualGraphMode v) {
  os << toString(v);
  return os;
}

std::string toString(RecomputationType r) {
  switch (r) {
  case RecomputationType::None:
    return "RecomputationType::None";
  case RecomputationType::Standard:
    return "RecomputationType::Standard";
  case RecomputationType::Pipeline:
    return "RecomputationType::Pipeline";
  case RecomputationType::NormOnly:
    return "RecomputationType::NormOnly";
  case RecomputationType::RecomputeAll:
    return "RecomputationType::RecomputeAll";
  case RecomputationType::N:
    throw error("Bad RecomputationType {}", static_cast<int>(r));
  default:
    throw error("Unknown RecomputationType");
  }
}

std::ostream &operator<<(std::ostream &os, RecomputationType r) {
  os << toString(r);
  return os;
}

SessionOptions::NumIOTiles::NumIOTiles() : value(0), userAssignedValue(false) {}
SessionOptions::NumIOTiles::NumIOTiles(int numIOTiles_)
    : value(numIOTiles_), userAssignedValue(true) {}

// Compare with ints.
bool SessionOptions::NumIOTiles::operator==(const int &rhs) const {
  int lhs = *this;
  return lhs == rhs;
}

// Auto convert to int.
SessionOptions::NumIOTiles::operator int() const {
  // If the option was set, it takes priority.
  if (userAssignedValue) {
    return value;
  } else {
    return 0;
  }
}

// Assign value using int.
SessionOptions::NumIOTiles &
SessionOptions::NumIOTiles::operator=(const int &x) {
  value             = x;
  userAssignedValue = true;
  return *this;
}

unsigned
SessionOptions::getPrefetchBufferingDepth(const TensorId &id,
                                          unsigned defaultValue) const {
  unsigned result = 1;
  if (enablePrefetchDatastreams) {
    auto mapIt = prefetchBufferingDepthMap.find(id);
    if (mapIt == prefetchBufferingDepthMap.end()) {
      result = defaultValue;
    } else {
      result = mapIt->second;
    }
  }
  if (result < 1) {
    throw error("Unable to support a buffering depth of {} for tensor {} "
                "(minimum buffering depth is 1)",
                result,
                id);
  }
  return result;
}

int64_t SessionOptions::getGlobalReplicationFactor() const {
  if (enableDistributedReplicatedGraphs) {
    return globalReplicationFactor;
  }
  if (enableReplicatedGraphs) {
    return replicatedGraphCount;
  }
  return 1LL;
}

unsigned SessionOptions::getAccumulationFactor() const {
  unsigned af = static_cast<unsigned>(accumulationFactor);
  if (!enableGradientAccumulation && static_cast<unsigned>(af) > 1) {
    // A check on user input consistency
    throw error(
        "enableGradientAccumulation is false, but accumulationFactor > 1. "
        "Either enable gradient accumulation, or set the accumulation factor "
        "to 1");
  }
  return af;
}

bool SessionOptions::autoRecomputationEnabled() const {
  return autoRecomputation != RecomputationType::None;
}

bool SessionOptions::shouldDelayVarUpdates() const {
  /*
    Delaying var updates only needed due to implicit recomputation hiding the
    liveness of the recomputed segments from the scheduler. If we have explicit
    recomputation and explicit main loops, the scheduler should have enough
    information to make the correct decisions without us changing the schedule
    priorities.
   */
  return delayVarUpdates && executionPhaseSettings.phases < 2 &&
         batchSerializationSettings.factor < 2 && !explicitRecomputation &&
         !enableExplicitMainLoops;
}

std::string toString(const TensorLocationSettings &tls) {
  std::stringstream ss;
  ss << "(";
  ss << "minElementsForOffChip=" << tls.minElementsForOffChip << ", ";
  ss << "minElementsForReplicatedTensorSharding="
     << tls.minElementsForReplicatedTensorSharding << ", ";
  ss << "location=" << tls.location;
  ss << ")";
  return ss.str();
}

std::ostream &operator<<(std::ostream &os, const TensorLocationSettings &tls) {
  os << toString(tls);
  return os;
}

// No implementation required

} // namespace popart

namespace std {
std::size_t hash<popart::SessionOptions>::operator()(
    const popart::SessionOptions &so) const {
  // Hash based on all the SessionOptions attributes that
  // can affect compiled program
  std::size_t seed = 0;

  boost::hash_combine(seed, so.rearrangeAnchorsOnHost);
  boost::hash_combine(seed, so.enableNonStableSoftmax);
  boost::hash_combine(seed, so.replicatedGraphCount);
  boost::hash_combine(seed, so.globalReplicationFactor);
  boost::hash_combine(seed, so.enablePipelining);
  boost::hash_combine(seed, so.accumulationFactor);
  boost::hash_combine(
      seed, static_cast<int>(so.accumulationAndReplicationReductionType));
  boost::hash_combine(
      seed,
      static_cast<int>(so.meanAccumulationAndReplicationReductionStrategy));
  boost::hash_combine(seed, so.enableFloatingPointChecks);
  boost::hash_combine(seed, so.enableStochasticRounding);
  boost::hash_combine(seed, so.enableFullyConnectedPass);
  boost::hash_combine(seed, static_cast<int>(so.syntheticDataMode));
  boost::hash_combine(seed, so.enableSerializedMatmuls);
  boost::hash_combine(seed, so.aliasZeroCopy);
  boost::hash_combine(seed, static_cast<int>(so.numIOTiles));
  boost::hash_combine(seed, so.enableOutlining);
  boost::hash_combine(seed, so.enableOutliningCopyCostPruning);
  boost::hash_combine(seed, so.outlineThreshold);
  boost::hash_combine(seed, so.outlineSequenceBreakCost);
  boost::hash_combine(seed, so.subgraphCopyingStrategy);
  boost::hash_combine(seed, static_cast<int>(so.autoRecomputation));
  boost::hash_combine(seed, static_cast<int>(so.mergeVarUpdate));
  boost::hash_combine(seed, so.mergeVarUpdateMemThreshold);
  boost::hash_combine(seed, so.looseThresholdAtPeak);
  boost::hash_combine(seed, so.explicitRecomputation);
  boost::hash_combine(seed, so.partialsTypeMatMuls);
  boost::hash_combine(seed, so.decomposeGradSum);
  boost::hash_combine(seed, static_cast<int>(so.virtualGraphMode));
  boost::hash_combine(seed, so.delayVarUpdates);
  boost::hash_combine(seed, so.scheduleNonWeightUpdateGradientConsumersEarly);
  boost::hash_combine(seed, so.enableStableNorm);
  boost::hash_combine(seed, so.timeLimitScheduler);
  boost::hash_combine(seed, so.swapLimitScheduler);
  boost::hash_combine(seed, so.groupHostSync);
  boost::hash_combine(seed, so.enableLoadAndOffloadRNGState);
  boost::hash_combine(seed, so.kahnTieBreaker);
  boost::hash_combine(seed, so.automaticLossScalingSettings.hash());
  boost::hash_combine(seed, so.enableSupportedDataTypeCasting);
  boost::hash_combine(seed, so.ensureFp32LossScaleTensor);

  boost::hash_combine(seed, so.groupNormStridedChannelGrouping);
  boost::hash_combine(
      seed, static_cast<int>(so.accumulateOuterFragmentSettings.schedule));
  boost::hash_range(
      seed,
      so.accumulateOuterFragmentSettings.excludedVirtualGraphs.begin(),
      so.accumulateOuterFragmentSettings.excludedVirtualGraphs.end());

  boost::hash_combine(seed, so.batchSerializationSettings.factor);
  boost::hash_combine(seed,
                      so.batchSerializationSettings.concatOnVirtualGraphChange);
  boost::hash_combine(
      seed, so.batchSerializationSettings.concatOnExecutionPhaseChange);
  boost::hash_combine(
      seed, so.batchSerializationSettings.concatOnPipelineStageChange);
  boost::hash_combine(
      seed, static_cast<int>(so.batchSerializationSettings.transformContext));
  boost::hash_combine(seed,
                      static_cast<int>(so.batchSerializationSettings.method));
  boost::hash_combine(
      seed, static_cast<int>(so.batchSerializationSettings.batchSchedule));

  boost::hash_combine(seed, so.autodiffSettings.stitchStrategy);

  boost::hash_combine(seed, so.executionPhaseSettings.phases);
  boost::hash_combine(seed, so.executionPhaseSettings.stages);
  boost::hash_combine(
      seed, static_cast<int>(so.executionPhaseSettings.weightIOSchedule));
  boost::hash_combine(
      seed, static_cast<int>(so.executionPhaseSettings.activationIOSchedule));
  boost::hash_combine(
      seed,
      static_cast<int>(so.executionPhaseSettings.optimizerStateIOSchedule));
  boost::hash_combine(
      seed, static_cast<int>(so.executionPhaseSettings.accumulatorIOSchedule));
  boost::hash_combine(seed,
                      static_cast<int>(so.executionPhaseSettings.schedule));

  boost::hash_combine(
      seed, so.activationTensorLocationSettings.minElementsForOffChip);
  boost::hash_combine(seed,
                      so.activationTensorLocationSettings
                          .minElementsForReplicatedTensorSharding);
  auto atls = so.activationTensorLocationSettings.location.serialize();
  boost::hash_range(seed, atls.begin(), atls.end());

  boost::hash_combine(seed,
                      so.weightTensorLocationSettings.minElementsForOffChip);
  boost::hash_combine(
      seed,
      so.weightTensorLocationSettings.minElementsForReplicatedTensorSharding);
  auto wtls = so.weightTensorLocationSettings.location.serialize();
  boost::hash_range(seed, wtls.begin(), wtls.end());

  boost::hash_combine(
      seed, so.optimizerStateTensorLocationSettings.minElementsForOffChip);
  boost::hash_combine(seed,
                      so.optimizerStateTensorLocationSettings
                          .minElementsForReplicatedTensorSharding);
  auto otls = so.optimizerStateTensorLocationSettings.location.serialize();
  boost::hash_range(seed, otls.begin(), otls.end());

  boost::hash_combine(
      seed, so.accumulatorTensorLocationSettings.minElementsForOffChip);
  boost::hash_combine(seed,
                      so.accumulatorTensorLocationSettings
                          .minElementsForReplicatedTensorSharding);
  auto acctls = so.accumulatorTensorLocationSettings.location.serialize();
  boost::hash_range(seed, acctls.begin(), acctls.end());

  for (auto key_val : so.engineOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }
  for (auto key_val : so.convolutionOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }
  for (auto key_val : so.lstmOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }
  for (auto key_val : so.matmulOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }
  for (auto key_val : so.gclOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }

  return seed;
}

} // namespace std
