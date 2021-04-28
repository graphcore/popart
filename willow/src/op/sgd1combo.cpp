// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

std::unique_ptr<Op> SGD1ComboOp::clone() const {
  return std::make_unique<SGD1ComboOp>(*this);
}

} // namespace popart
