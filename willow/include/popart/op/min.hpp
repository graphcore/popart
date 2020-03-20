// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MIN_HPP
#define GUARD_NEURALNET_MIN_HPP

#include <popart/op/variadic.hpp>

namespace popart {

class MinOp : public VariadicOp {
public:
  MinOp(const OperatorIdentifier &_opid, const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class MinArgGradOp : public NonLinearVariadicGradOp {
public:
  MinArgGradOp(const MinOp &, InIndex);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
