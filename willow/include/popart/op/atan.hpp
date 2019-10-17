#ifndef GUARD_NEURALNET_ATAN_HPP
#define GUARD_NEURALNET_ATAN_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class AtanOp : public ElementWiseUnaryOp {
public:
  AtanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class AtanInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  AtanInplaceOp(const AtanOp &);
  std::unique_ptr<Op> clone() const final;
};

class AtanGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  AtanGradOp(const AtanOp &);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
