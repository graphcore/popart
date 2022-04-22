// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cstdlib>
#include <type_traits>
#include <popart/version.hpp>

namespace {
template <typename T> struct is_string_literal : std::false_type {};
template <std::size_t N>
struct is_string_literal<const char (&)[N]> : std::true_type {};

template <typename T>
constexpr bool is_string_literal_v = is_string_literal<T>::value;
} // namespace

namespace popart {
namespace core {
const char *versionString() {
  // We won't validate the version string here.
  static_assert(is_string_literal_v<decltype(POPART_VERSION_STRING_WITH_HASH)>,
                "POPART_VERSION_STRING_WITH_HASH must be a string literal");
  static_assert(
      std::extent<decltype(POPART_VERSION_STRING_WITH_HASH)>::value >=
          std::extent<decltype("1.2.3 (1234)")>::value,
      "POPART_VERSION_STRING_WITH_HASH must have at least 12 characters");
  return POPART_VERSION_STRING_WITH_HASH;
}

const char *packageHash() {
  static_assert(is_string_literal_v<decltype(POPART_PACKAGE_HASH)>,
                "POPART_PACKAGE_HASH must be a string literal");
  return POPART_PACKAGE_HASH;
}

VersionNumber versionNumber() { return versionGetCurrent(); }

const char *versionHashString() {
  static_assert(is_string_literal_v<decltype(POPART_VERSION_HASH)>,
                "POPART_VERSION_HASH must be a string literal");
  return POPART_VERSION_HASH;
}

static_assert(VersionNumber{1, 10, 10} < VersionNumber{2, 0, 2},
              "Version 1.10.10 must precede 2.0.2");

static_assert(VersionNumber{1, 0, 0} < VersionNumber{1, 1, 0},
              "Version 1.0.0 must precede 1.1.0");

static_assert(VersionNumber{1, 0, 0} < VersionNumber{1, 0, 1},
              "Version 1.0.0 must precede 1.0.1");

static_assert(!(VersionNumber{2, 0, 0} < VersionNumber{1, 0, 1}),
              "Version 2.0.0 must not precede 1.0.1");

static_assert(VersionNumber{1, 10, 10} == VersionNumber{1, 10, 10},
              "Version 1.10.10 must be equal to 1.10.10");

static_assert(VersionNumber{1, 10, 10} != VersionNumber{1, 10, 11},
              "Version 1.10.10 must not be equal to 1.10.11");

} // namespace core
} // namespace popart
