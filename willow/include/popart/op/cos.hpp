#ifndef GUARD_NEURALNET_COS_HPP
#define GUARD_NEURALNET_COS_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class CosOp : public ElementWiseUnaryOp {
public:
  CosOp(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static OperatorIdentifier getOpId(const Ir &ir);
};

class CosGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  CosGradOp(const CosOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
