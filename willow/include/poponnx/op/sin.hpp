#ifndef GUARD_NEURALNET_SIN_HPP
#define GUARD_NEURALNET_SIN_HPP

#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class SinOp : public ElementWiseUnaryOp {
public:
  SinOp(const OpConstructorBundle &);
  SinOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SinGradOp : public ElementWiseNonLinearUnaryGradOp {
public:
  SinGradOp(SinOp *fwdOp);
  std::unique_ptr<Op> clone() const final;
};

} // namespace poponnx

#endif
