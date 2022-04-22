// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SIGN_HPP
#define GUARD_NEURALNET_SIGN_HPP

#include <memory>
#include <popart/op/onewayunary.hpp>

#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {
class Ir;

class SignOp : public OneWayUnaryOp {
public:
  SignOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;

  static OperatorIdentifier getOpId(const Ir &ir);

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

class SignInplaceOp : public OneWayUnaryInPlaceOp {
public:
  SignInplaceOp(const SignOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
