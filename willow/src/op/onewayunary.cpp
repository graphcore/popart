
// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <popart/op/onewayunary.hpp>
#include <popart/op/zeros.hpp>
#include <popart/opmanager.hpp>

namespace popart {

OneWayUnaryOp::OneWayUnaryOp(const OperatorIdentifier &_opid,
                             const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> OneWayUnaryOp::getGradOps() {
  return UnaryZeroGradOp::getGradOpVector(getSettings());
}

OneWayUnaryInPlaceOp::OneWayUnaryInPlaceOp(const OperatorIdentifier &_opid,
                                           const Op::Settings &settings_)
    : ElementWiseInplaceUnaryOp(_opid, settings_) {}

} // namespace popart
