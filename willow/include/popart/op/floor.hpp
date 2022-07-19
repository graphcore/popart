// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_FLOOR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_FLOOR_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/onewayunary.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class FloorOp : public OneWayUnaryOp {
public:
  FloorOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class FloorInplaceOp : public OneWayUnaryInPlaceOp {
public:
  FloorInplaceOp(const FloorOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_FLOOR_HPP_
