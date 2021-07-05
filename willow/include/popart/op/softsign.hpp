// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTSIGN_HPP
#define GUARD_NEURALNET_SOFTSIGN_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class SoftSignOp : public ElementWiseUnaryOp {
public:
  SoftSignOp(const OperatorIdentifier &opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class SoftSignInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SoftSignInplaceOp(const SoftSignOp &);
  std::unique_ptr<Op> clone() const final;
};

class SoftSignGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SoftSignGradOp(const SoftSignOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
