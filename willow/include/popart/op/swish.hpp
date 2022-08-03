// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SWISH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SWISH_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class SwishOp : public ElementWiseUnaryOp {
public:
  SwishOp(const OperatorIdentifier &opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SwishInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SwishInplaceOp(const SwishOp &);
  SwishInplaceOp(const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
};

class SwishGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SwishGradOp(const SwishOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SWISH_HPP_
