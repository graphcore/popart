// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/op/sgd2acclupdate.hpp>

#include <memory>

namespace popart {
std::unique_ptr<Op> SGD2PartialAcclUpdateOp::clone() const {
  return std::make_unique<SGD2PartialAcclUpdateOp>(*this);
}

} // namespace popart
