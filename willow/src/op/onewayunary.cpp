
// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popart/op/onewayunary.hpp>
#include <popart/op/zeros.hpp>

#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

OneWayUnaryOp::OneWayUnaryOp(const OperatorIdentifier &_opid,
                             const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> OneWayUnaryOp::clone() const {
  return std::make_unique<OneWayUnaryOp>(*this);
}

std::vector<std::unique_ptr<Op>> OneWayUnaryOp::getGradOps() {
  return UnaryZeroGradOp::getGradOpVector(getSettings());
}

OneWayUnaryInPlaceOp::OneWayUnaryInPlaceOp(const OperatorIdentifier &_opid,
                                           const Op::Settings &settings_)
    : ElementWiseInplaceUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> OneWayUnaryInPlaceOp::clone() const {
  return std::make_unique<OneWayUnaryInPlaceOp>(*this);
}

} // namespace popart
