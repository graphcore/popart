// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ROUND_HPP
#define GUARD_NEURALNET_ROUND_HPP

#include <popart/op/onewayunary.hpp>

namespace popart {

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

#endif
