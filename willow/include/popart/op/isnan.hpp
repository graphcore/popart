#ifndef GUARD_NEURALNET_ISNAN_HPP
#define GUARD_NEURALNET_ISNAN_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class IsNaN : public ElementWiseUnaryBooleanOp {
public:
  IsNaN(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  static OperatorIdentifier getOpId(const Ir &ir);
};

} // namespace popart

#endif
