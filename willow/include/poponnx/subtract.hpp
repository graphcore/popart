#ifndef GUARD_NEURALNET_SUBTRACT_HPP
#define GUARD_NEURALNET_SUBTRACT_HPP

#include <poponnx/ir.hpp>

namespace willow {

class SubtractOp : public Op {
public:
  SubtractOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

class SubtractGradOp : public Op {
public:
  SubtractGradOp(SubtractOp *);
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  // Info on Tensors 0 and 1.
  // gradient of an input has the same
  // shape and type as the input itself
  TensorInfo info0;
  TensorInfo info1;
};

} // namespace willow

#endif
