#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {
class LogSoftmaxOp : public Op {
public:
  LogSoftmaxOp(const onnx::NodeProto &node, Graph *pgraph);

  virtual void setup() override final;
};


// if p = lsm(v) 
//

class LogSoftmaxGradOp : public GradOp {

public:
  LogSoftmaxGradOp(LogSoftmaxOp *);
  virtual Op *getNonGradOp() override final;
  virtual const std::vector<GradInOutMapper> &
  gradInputInfo() const override final;
  virtual const std::map<int, int> &gradOutToNonGradIn() const override final;
  virtual void setup() override final;

private:
  std::vector<GradInOutMapper> createLogSoftmaxGradInfo() const;
  std::map<int, int> createLogSoftmaxGradOutToIn() const;
  LogSoftmaxOp *logsoftmaxOp;
};


} // namespace neuralnet

#endif
