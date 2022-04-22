// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/boundary.hpp>

#include "popart/op.hpp"

namespace popart {
std::unique_ptr<Op> BoundaryOp::clone() const {
  return std::make_unique<BoundaryOp>(*this);
}
} // namespace popart
