// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOG1P_HPP
#define GUARD_NEURALNET_LOG1P_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

// Compute log(x + 1).

class Log1pOp : public ElementWiseUnaryOp {
public:
  Log1pOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class Log1pInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  Log1pInplaceOp(const Log1pOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
