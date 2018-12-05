#ifndef GUARD_NEURALNET_SIN_HPP
#define GUARD_NEURALNET_SIN_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class SinOp : public Op {
public:
  SinOp(const OpConstructorBundle &);
  SinOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

class SinGradOp : public Op {
public:
  SinGradOp(SinOp *fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getFwdArgInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }
};

} // namespace poponnx

#endif
