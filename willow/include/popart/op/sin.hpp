#ifndef GUARD_NEURALNET_SIN_HPP
#define GUARD_NEURALNET_SIN_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class SinOp : public ElementWiseUnaryOp {
public:
  SinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SinGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SinGradOp(const SinOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
