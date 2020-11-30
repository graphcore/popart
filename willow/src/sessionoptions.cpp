// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>

#include <boost/lexical_cast.hpp>

#include <popart/error.hpp>
#include <popart/sessionoptions.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

constexpr static int NDotChecks = static_cast<int>(DotCheck::N);

namespace {

std::array<std::string, NDotChecks> getDotCheckIds() {

  std::array<std::string, NDotChecks> V;
  V[static_cast<int>(DotCheck::Fwd0)]     = "fwd0";
  V[static_cast<int>(DotCheck::Fwd1)]     = "fwd1";
  V[static_cast<int>(DotCheck::Bwd0)]     = "bwd0";
  V[static_cast<int>(DotCheck::PreAlias)] = "prealias";
  V[static_cast<int>(DotCheck::Final)]    = "final";

  // verify that we have registered all the DotChecks
  // c++, when will we be able to make this constexpr?
  if (!std::all_of(V.cbegin(), V.cend(), [](const std::string &s) {
        return s.size() > 0;
      })) {
    throw error("Not all DotChecks have a string registered in getDotCheckIds");
  }

  return V;
}

} // namespace

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

std::string getDotCheckString(DotCheck d) {
  const static std::array<std::string, NDotChecks> V = getDotCheckIds();
  return V[static_cast<int>(d)];
}

DotCheck dotCheckFromString(const std::string &s) {
  const static std::map<std::string, DotCheck> dotCheckStringsMap{
      {"FWD0", DotCheck::Fwd0},
      {"FWD1", DotCheck::Fwd1},
      {"BWD0", DotCheck::Bwd0},
      {"PREALIAS", DotCheck::PreAlias},
      {"FINAL", DotCheck::Final}};

  auto found = dotCheckStringsMap.find(s);
  if (found != dotCheckStringsMap.end()) {
    return found->second;
  } else {
    throw error("Unrecognised dot check '{}'", s);
  }
}

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
  }
  // The GCL environment variable should only be used if the session option has
  // not been set.
  else if (std::getenv("GCL_NUM_IO_TILES")) {
    logging::warn(
        "You are using a deprecated environement variable \"{}\". This "
        "will be removed in an upcoming release. Please use the "
        "session option 'SessionOptions::{}' instead",
        "GCL_NUM_IO_TILES",
        "numIOTiles");

    const char *env_p = std::getenv("GCL_NUM_IO_TILES");
    return boost::lexical_cast<int>(env_p);
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

unsigned SessionOptions::getPrefetchBufferingDepth(const TensorId &id) const {
  if (!enablePrefetchDatastreams) {
    return 1;
  } else {
    auto mapIt = prefetchBufferingDepthMap.find(id);
    if (mapIt == prefetchBufferingDepthMap.end()) {
      return 1;
    } else {
      auto bufferingDepth = mapIt->second;
      if (bufferingDepth < 1) {
        throw error("Unable to support a buffering depth of {} for tensor {} "
                    "(minimum buffering depth is 1)",
                    bufferingDepth,
                    id);
      }
      return bufferingDepth;
    }
  }
}

// No implementation required

} // namespace popart

namespace std {
std::size_t hash<popart::SessionOptions>::operator()(
    const popart::SessionOptions &so) const {
  // Hash based on all the SessionOptions attributes that
  // can affect compiled program

  std::stringstream ss;
  ss << so.autoRecomputation;

  std::size_t seed = std::hash<std::string>{}(ss.str());

  boost::hash_combine(seed, so.rearrangeAnchorsOnHost);
  boost::hash_combine(seed, so.enableNonStableSoftmax);
  boost::hash_combine(seed, so.replicatedGraphCount);
  boost::hash_combine(seed, so.enablePipelining);
  boost::hash_combine(seed, so.enableFloatingPointChecks);
  boost::hash_combine(seed, so.enableStochasticRounding);
  boost::hash_combine(seed, so.enableFullyConnectedPass);
  boost::hash_combine(seed, static_cast<int>(so.syntheticDataMode));
  boost::hash_combine(seed, so.enableSerializedMatmuls);
  boost::hash_combine(seed, so.enableGroupedMatmuls);
  boost::hash_combine(seed, so.aliasZeroCopy);
  boost::hash_combine(seed, static_cast<int>(so.numIOTiles));
  boost::hash_combine(seed, so.enableStochasticRounding);
  boost::hash_combine(seed, so.enableStochasticRounding);
  boost::hash_combine(seed, static_cast<int>(so.autoRecomputation));
  boost::hash_combine(seed, static_cast<int>(so.mergeVarUpdate));
  boost::hash_combine(seed, so.partialsTypeMatMuls);
  boost::hash_combine(seed, so.decomposeGradSum);
  boost::hash_combine(seed, static_cast<int>(so.virtualGraphMode));
  boost::hash_combine(seed, static_cast<int>(so.syntheticDataMode));

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
  for (auto key_val : so.gclOptions) {
    boost::hash_combine(seed, key_val.first);
    boost::hash_combine(seed, key_val.second);
  }

  return seed;
}

} // namespace std
