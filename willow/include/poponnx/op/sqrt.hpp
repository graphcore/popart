#ifndef GUARD_NEURALNET_SQRT_HPP
#define GUARD_NEURALNET_SQRT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// y = sqrt(x)
class SqrtOp : public Op {
public:
  SqrtOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
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
