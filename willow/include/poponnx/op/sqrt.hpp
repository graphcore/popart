#ifndef GUARD_NEURALNET_SQRT_HPP
#define GUARD_NEURALNET_SQRT_HPP

#include <poponnx/op.hpp>
#include <poponnx/op/elementwise.hpp>

namespace poponnx {

// y = sqrt(x)
class SqrtOp : public ElementWiseUnaryOp {
public:
  SqrtOp(const OperatorIdentifier &_opid,
         Ir *_ir,
         const std::string &name = "",
         const Attributes &_attr = {});
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
};

class SqrtGradOp : public Op {
public:
  SqrtGradOp(SqrtOp *fwdOp);
  std::unique_ptr<Op> clone() const final;
  void setup() final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdOutInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
