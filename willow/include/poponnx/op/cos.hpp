#ifndef GUARD_NEURALNET_COS_HPP
#define GUARD_NEURALNET_COS_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class CosOp : public ElementWiseUnaryOp {
public:
  CosOp(const OperatorIdentifier &_opid, const Op::Settings &);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class CosGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  CosGradOp(const CosOp &fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
