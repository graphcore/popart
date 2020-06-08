// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/basicoptionals.hpp>
#include <popart/error.hpp>

namespace popart {
[[noreturn]] void noValueBasicOptionalError() {
  throw error("No value set for this BasicOptional, cannot dereference it");
}
} // namespace popart
