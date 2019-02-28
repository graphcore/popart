#ifndef GUARD_NEURALNET_SUM_HPP
#define GUARD_NEURALNET_SUM_HPP

#include <poponnx/op/variadic.hpp>

namespace poponnx {

class SumOp : public VariadicOp {
public:
  SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class SumArgGradOp : public LinearVariadicGradOp {
public:
  SumArgGradOp(const SumOp &, InIndex inIndex);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
};

} // namespace poponnx

#endif
