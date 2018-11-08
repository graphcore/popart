#ifndef GUARD_NEURALNET_SQUEEZE_HPP
#define GUARD_NEURALNET_SQUEEZE_HPP

#include <willow/ir.hpp>

namespace willow {

class SqueezeOp : public Op {
public:
  SqueezeOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  virtual std::unique_ptr<Op> clone() const override final;
};

class SqueezeGradOp : public Op {
public:
  SqueezeGradOp(SqueezeOp *);
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createSqueezeGradInfo() const;
  std::map<int, int> createSqueezeGradOutToIn() const;
  // The shape and type of the input to the constructing forward op
  TensorInfo unsqueezedInfo;
};

} // namespace willow

#endif
