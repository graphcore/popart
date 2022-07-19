// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ROUND_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ROUND_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/onewayunary.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class RoundOp : public OneWayUnaryOp {
public:
  RoundOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class RoundInplaceOp : public OneWayUnaryInPlaceOp {
public:
  RoundInplaceOp(const RoundOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ROUND_HPP_
