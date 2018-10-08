#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <willow/graph.hpp>

namespace willow {
class LogSoftmaxOp : public Op {
public:
  LogSoftmaxOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual std::unique_ptr<Op> clone() const override final;
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
};

class LogSoftmaxGradOp : public GradOp {
public:
  LogSoftmaxGradOp(LogSoftmaxOp *);
  virtual Op *getNonGradCreator() const override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createLogSoftmaxGradInfo() const;
  std::map<int, int> createLogSoftmaxGradOutToIn() const;
  LogSoftmaxOp *logsoftmaxOp;
};

} // namespace willow

#endif
