// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ONEWAYUNARY_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ONEWAYUNARY_HPP_

#include <memory>
#include <vector>
#include <popart/op/elementwise.hpp>

#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

// A unary op which always returns a zero gradient

class OneWayUnaryOp : public ElementWiseUnaryOp {
public:
  OneWayUnaryOp(const OperatorIdentifier &, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class OneWayUnaryInPlaceOp : public ElementWiseInplaceUnaryOp {
public:
  OneWayUnaryInPlaceOp(const OperatorIdentifier &, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ONEWAYUNARY_HPP_
