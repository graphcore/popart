#ifndef GUARD_NEURALNET_NEGATE_HPP
#define GUARD_NEURALNET_NEGATE_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

class NegateOp : public ElementWiseUnaryOp {
public:
  NegateOp(const OperatorIdentifier &_opid,
           Ir *_ir,
           const std::string &name = "",
           const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class NegateGradOp : public NegateOp {
public:
  NegateGradOp(NegateOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
