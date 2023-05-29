// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NEARBYINT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NEARBYINT_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/onewayunary.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class NearbyIntOp : public OneWayUnaryOp {
public:
  using OneWayUnaryOp::OneWayUnaryOp;

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class NearbyIntInplaceOp : public OneWayUnaryInPlaceOp {
public:
  NearbyIntInplaceOp(const NearbyIntOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NEARBYINT_HPP_
