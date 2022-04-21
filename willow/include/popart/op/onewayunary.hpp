// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONEWAYUNARY_HPP
#define GUARD_NEURALNET_ONEWAYUNARY_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

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

#endif
