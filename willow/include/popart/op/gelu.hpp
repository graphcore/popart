// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GELU_HPP
#define GUARD_NEURALNET_GELU_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class GeluOp : public ElementWiseUnaryOp {
public:
  GeluOp(const OperatorIdentifier &opid, const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;
  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class GeluInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  GeluInplaceOp(const GeluOp &);
  std::unique_ptr<Op> clone() const final;
};

class GeluGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  GeluGradOp(const GeluOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
