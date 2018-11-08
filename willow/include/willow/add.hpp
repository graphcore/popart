#ifndef GUARD_NEURALNET_ADD_HPP
#define GUARD_NEURALNET_ADD_HPP

#include <willow/ir.hpp>

namespace willow {

class AddOp : public Op {
public:
  AddOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

class AddGradOp : public Op {
public:
  AddGradOp(AddOp *);
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createAddGradInfo() const;
  std::map<int, int> createAddGradOutToIn() const;
  // Info on Tensors 0 and 1.
  // gradient of an input has the same
  // shape and type as the input itself
  TensorInfo info0;
  TensorInfo info1;
};

} // namespace willow

#endif
