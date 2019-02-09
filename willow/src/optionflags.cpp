#include <algorithm>
#include <array>
#include <poponnx/error.hpp>
#include <poponnx/optionflags.hpp>

namespace poponnx {

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

// No implementation required

} // namespace poponnx
