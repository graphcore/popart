// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SINH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SINH_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class SinhOp : public ElementWiseUnaryOp {
public:
  SinhOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SinhInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SinhInplaceOp(const SinhOp &);
  SinhInplaceOp(const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
};

class SinhGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SinhGradOp(const SinhOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SINH_HPP_
