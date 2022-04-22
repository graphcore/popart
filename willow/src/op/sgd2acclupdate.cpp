// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <popart/op/sgd2acclupdate.hpp>

#include "popart/op.hpp"

namespace popart {
std::unique_ptr<Op> SGD2PartialAcclUpdateOp::clone() const {
  return std::make_unique<SGD2PartialAcclUpdateOp>(*this);
}

} // namespace popart
