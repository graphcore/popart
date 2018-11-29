#ifndef GUARD_NEURALNET_IDENTITY_HPP
#define GUARD_NEURALNET_IDENTITY_HPP

#include <poponnx/ir.hpp>

namespace poponnx {

class IdentityOp : public Op {
public:
  IdentityOp(const OpConstructorBundle &);
  IdentityOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
};

class IdentityGradOp : public IdentityOp {
public:
  IdentityGradOp(IdentityOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
};

} // namespace poponnx

#endif
