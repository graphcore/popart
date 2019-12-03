#ifndef GUARD_NEURALNET_MEAN_HPP
#define GUARD_NEURALNET_MEAN_HPP

#include <popart/op/variadic.hpp>

namespace popart {

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
  std::unique_ptr<Op> clone() const final;

  bool hasScale() const final { return true; }
  float getScale() const final { return 1.0f / static_cast<float>(nInputs); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  std::vector<GradInOutMapper> gradInputInfoVec;
  int nInputs;
};

} // namespace popart

#endif
