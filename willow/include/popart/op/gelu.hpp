// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_GELU_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_GELU_HPP_

#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

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
  GeluInplaceOp(const Op::Settings &opSettings);

  std::unique_ptr<Op> clone() const final;
};

class GeluGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  GeluGradOp(const GeluOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_GELU_HPP_
