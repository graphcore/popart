#ifndef GUARD_NEURALNET_TAN_HPP
#define GUARD_NEURALNET_TAN_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class TanOp : public ElementWiseUnaryOp {
public:
  TanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
