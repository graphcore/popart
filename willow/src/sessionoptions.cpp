// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <array>

#include <boost/lexical_cast.hpp>

#include <popart/error.hpp>
#include <popart/sessionoptions.hpp>

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
    BatchSerializationBatchSchedule batchSchedule_)
    : factor{factor_}, concatOnVirtualGraphChange{concatOnVirtualGraphChange_},
      concatOnExecutionPhaseChange{concatOnExecutionPhaseChange_},
      concatOnPipelineStageChange{concatOnPipelineStageChange_},
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

// No implementation required

} // namespace popart

namespace std {
std::size_t hash<popart::SessionOptions>::operator()(
    const popart::SessionOptions &so) const {
  // Hash based on all the SessionOptions attributes that
  // can affect compiled program

  std::stringstream ss;
  ss << so.autoRecomputation;

  auto hsh = std::hash<std::string>{}(ss.str());
  hsh      = (hsh ^ (std::hash<bool>{}(so.rearrangeAnchorsOnHost) << 1)) << 1;
  hsh      = (hsh ^ (std::hash<bool>{}(so.enableNonStableSoftmax) << 1)) << 1;
  hsh      = (hsh ^ (std::hash<int64_t>{}(so.replicatedGraphCount) << 1)) << 1;
  hsh      = (hsh ^ (std::hash<bool>{}(so.enablePipelining) << 1)) << 1;
  hsh = (hsh ^ (std::hash<bool>{}(so.enableFloatingPointChecks) << 1)) << 1;
  hsh = (hsh ^ (std::hash<bool>{}(so.enableStochasticRounding) << 1)) << 1;
  hsh = (hsh ^ (std::hash<bool>{}(so.enableFullyConnectedPass) << 1)) << 1;
  hsh = (hsh ^ (std::hash<int>{}(static_cast<int>(so.syntheticDataMode)) << 1))
        << 1;
  for (auto key_val : so.engineOptions) {
    hsh = (hsh ^ (std::hash<std::string>()(key_val.first) << 1)) << 1;
    hsh = (hsh ^ (std::hash<std::string>()(key_val.second) << 1)) << 1;
  }
  for (auto key_val : so.convolutionOptions) {
    hsh = (hsh ^ (std::hash<std::string>()(key_val.first) << 1)) << 1;
    hsh = (hsh ^ (std::hash<std::string>()(key_val.second) << 1)) << 1;
  }

  return hsh;
}
} // namespace std
