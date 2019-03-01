#ifndef GUARD_NEURALNET_MEAN_HPP
#define GUARD_NEURALNET_MEAN_HPP

#include <poponnx/op/variadic.hpp>

namespace poponnx {

class MeanOp : public VariadicOp {
public:
  MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;

private:
  virtual std::unique_ptr<Op> getIthGrad(int) const final;
};

class MeanArgGradOp : public LinearVariadicGradOp {
public:
  MeanArgGradOp(const MeanOp &, InIndex inIndex);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;

  bool hasScale() const final { return true; }
  float getScale() const final { return 1.0f / static_cast<float>(nInputs); }

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
  int nInputs;
};

} // namespace poponnx

#endif
