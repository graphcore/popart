// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popart/op/accumulatorzero.hpp"

namespace popart {

std::unique_ptr<Op> AccumulatorZeroOp::clone() const {
  return std::make_unique<AccumulatorZeroOp>(*this);
}

} // namespace popart
