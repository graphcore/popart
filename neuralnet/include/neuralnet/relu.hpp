#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class ReluOp : public Op {
public:
  ReluOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

// takes output of ReluOp as input and not the output of ReluOp
// to determine where gradients become zero. It might be better
// (depending in what can be in-placed) to rather take the output
// of ReluOp in to do this (or another binary tensor). This is why
// I have called it ""
class ReluGradOp : public GradOp {
public:
  ReluGradOp(ReluOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createReluGradInfo() const;
  std::map<int, int> createReluGradOutToIn() const;
  ReluOp *reluOp;
};

} // namespace neuralnet

#endif
