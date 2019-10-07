#include <algorithm>
#include <array>
#include <popart/error.hpp>
#include <popart/optionflags.hpp>

namespace popart {

constexpr static int NDotChecks = static_cast<int>(DotCheck::N);

namespace {

std::array<std::string, NDotChecks> getDotCheckIds() {

  std::array<std::string, NDotChecks> V;
  V[static_cast<int>(DotCheck::FWD0)]     = "fwd0";
  V[static_cast<int>(DotCheck::FWD1)]     = "fwd1";
  V[static_cast<int>(DotCheck::BWD0)]     = "bwd0";
  V[static_cast<int>(DotCheck::PREALIAS)] = "prealias";
  V[static_cast<int>(DotCheck::FINAL)]    = "final";

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

std::string getDotCheckString(DotCheck d) {
  const static std::array<std::string, NDotChecks> V = getDotCheckIds();
  return V[static_cast<int>(d)];
}

DotCheck dotCheckFromString(const std::string &s) {
  const static std::map<std::string, DotCheck> dotCheckStringsMap{
      {"FWD0", DotCheck::FWD0},
      {"FWD1", DotCheck::FWD1},
      {"BWD0", DotCheck::BWD0},
      {"PREALIAS", DotCheck::PREALIAS},
      {"FINAL", DotCheck::FINAL}};

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
    return "VirtualGraphMode::Off";
  case RecomputationType::Standard:
    return "VirtualGraphMode::Manual";
  case RecomputationType::NormOnly:
    return "VirtualGraphMode::Auto";
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

// No implementation required

} // namespace popart
