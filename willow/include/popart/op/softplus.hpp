// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTPLUS_HPP
#define GUARD_NEURALNET_SOFTPLUS_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class SoftPlusOp : public ElementWiseUnaryOp {
public:
  SoftPlusOp(const OperatorIdentifier &opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SoftPlusInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SoftPlusInplaceOp(const SoftPlusOp &);
  std::unique_ptr<Op> clone() const final;
};

class SoftPlusGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SoftPlusGradOp(const SoftPlusOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
