// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <memory>
#include <popart/op/sgd2varupdate.hpp>

#include "popart/op.hpp"

namespace popart {

std::unique_ptr<Op> SGD2VarUpdateOp::clone() const {
  return std::make_unique<SGD2VarUpdateOp>(*this);
}

} // namespace popart
