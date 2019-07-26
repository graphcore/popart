#ifndef GUARD_NEURALNET_SQUARE_HPP
#define GUARD_NEURALNET_SQUARE_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class SquareOp : public ElementWiseUnaryOp {
public:
  SquareOp(const OperatorIdentifier &_opid, const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

} // namespace popart

#endif
