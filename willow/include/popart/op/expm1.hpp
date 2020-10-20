// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXPM1_HPP
#define GUARD_NEURALNET_EXPM1_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

// Compute exp(x) - 1.

class Expm1Op : public ElementWiseUnaryOp {
public:
  Expm1Op(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class Expm1InplaceOp : public ElementWiseInplaceUnaryOp {
public:
  Expm1InplaceOp(const Expm1Op &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
