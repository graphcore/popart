#ifndef GUARD_NEURALNET_SQUEEZE_HPP
#define GUARD_NEURALNET_SQUEEZE_HPP

#include <poponnx/ir.hpp>

namespace willow {

class SqueezeOp : public Op {
public:
  SqueezeOp(const onnx::NodeProto &node, Ir *pir);
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;
};

class SqueezeGradOp : public Op {
public:
  SqueezeGradOp(SqueezeOp *);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

private:
  // The shape and type of the input to the constructing forward op
  TensorInfo unsqueezedInfo;
};

} // namespace willow

#endif
