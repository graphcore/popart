// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOG1P_HPP
#define GUARD_NEURALNET_LOG1P_HPP

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

// Compute log(x + 1).

class Log1pOp : public ElementWiseUnaryOp {
public:
  Log1pOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class Log1pInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  Log1pInplaceOp(const Log1pOp &);
  std::unique_ptr<Op> clone() const final;
};

class Log1pGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  Log1pGradOp(const Log1pOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
