#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <poponnx/ir.hpp>

namespace willow {

class ReluOp : public Op {
public:
  ReluOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

// takes output of ReluOp as input and not the input of ReluOp
// to determine where gradients become zero. It might be better
// (depending in what can be in-placed) to rather take the input
// of ReluOp in to do this (or a boolean tensor).
class ReluGradOp : public Op {
public:
  ReluGradOp(ReluOp *);
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

  // The index at which the output of the Relu (the "relud" tensor)
  // is an input to this ReluGradOp
  int getReludIn() const;

  // The index at which the gradient of the output of
  // the Relu is an input to this ReluGradOp
  int getGradReludIn() const;
};

} // namespace willow

#endif
