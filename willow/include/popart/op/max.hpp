#ifndef GUARD_NEURALNET_MAX_HPP
#define GUARD_NEURALNET_MAX_HPP

#include <popart/op/variadic.hpp>

namespace popart {

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

#endif
