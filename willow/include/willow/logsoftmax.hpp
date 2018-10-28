#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <willow/ir.hpp>

namespace willow {
class LogSoftmaxOp : public Op {
public:
  LogSoftmaxOp(const onnx::NodeProto &node, Ir *pir);
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

// not a gradient of a single Op, so not inheriting from GradOp
class LogSoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a LogSoftMaxOp
  LogSoftmaxGradDirectOp(Op *);
  virtual std::unique_ptr<Op> clone() const override final;
  // this Op has no Grad Ops, throw error if called
  virtual std::vector<std::unique_ptr<Op>> getGradOps() override final;
  virtual void setup() override final;
  LogSoftmaxOp *getLogSofmaxOp() const;

private:
  LogSoftmaxOp *logsoftmaxOp;
};

} // namespace willow

#endif
