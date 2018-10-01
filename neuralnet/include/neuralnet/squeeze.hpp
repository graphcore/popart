#ifndef GUARD_NEURALNET_SQUEEZE_HPP
#define GUARD_NEURALNET_SQUEEZE_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class SqueezeOp : public Op {
public:
  SqueezeOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;

  virtual void setup() override final;
  virtual std::unique_ptr<Op> clone() const override final;
};

class SqueezeGradOp : public GradOp {

public:
  SqueezeGradOp(SqueezeOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createSqueezeGradInfo() const;
  std::map<int, int> createSqueezeGradOutToIn() const;
  SqueezeOp *squeezeOp;
};

} // namespace neuralnet

#endif
