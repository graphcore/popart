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
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class Expm1InplaceOp : public ElementWiseInplaceUnaryOp {
public:
  Expm1InplaceOp(const Expm1Op &);
  std::unique_ptr<Op> clone() const final;
};

// Note that Expm1GradOp does NOT
// follow the pattern of ElementWiseNonLinearUnaryGradOp
// because it takes the output of Expm1 as an input, and does
// not take the input of Expm1 as an input.
class Expm1GradOp : public Op {
public:
  Expm1GradOp(const Expm1Op &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }

  // The input index to this Op of the output of the Expm1
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
