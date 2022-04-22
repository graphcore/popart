// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/placeholder.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

PlaceholderOp::PlaceholderOp(const OperatorIdentifier &opid_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_) {}

std::unique_ptr<Op> PlaceholderOp::clone() const {
  return std::make_unique<PlaceholderOp>(*this);
}

} // namespace popart
