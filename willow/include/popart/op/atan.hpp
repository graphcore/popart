// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ATAN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ATAN_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class AtanOp : public ElementWiseUnaryOp {
public:
  AtanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class AtanInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  AtanInplaceOp(const AtanOp &);
  std::unique_ptr<Op> clone() const final;
};

class AtanGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  AtanGradOp(const AtanOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ATAN_HPP_
