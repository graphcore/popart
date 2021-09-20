// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SWISH_HPP
#define GUARD_NEURALNET_SWISH_HPP

#include <memory>

#include <popart/op/elementwise.hpp>

namespace popart {

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
  std::unique_ptr<Op> clone() const final;
};

class SwishGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SwishGradOp(const SwishOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
