// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_MAX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_MAX_HPP_

#include <memory>
#include <popart/op/variadic.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
struct OperatorIdentifier;

class MaxOp : public VariadicOp {
public:
  MaxOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class MaxArgGradOp : public NonLinearVariadicGradOp {
public:
  MaxArgGradOp(const MaxOp &, InIndex);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_MAX_HPP_
