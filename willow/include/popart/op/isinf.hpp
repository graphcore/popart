#ifndef GUARD_NEURALNET_ISINF_HPP
#define GUARD_NEURALNET_ISINF_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class IsInf : public ElementWiseUnaryBooleanOp {
public:
  IsInf(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;

  static OperatorIdentifier getOpId(const Ir &ir);
};

} // namespace popart

#endif
