#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {
class LogSoftmaxOp : public NonGradOp {
public:
  LogSoftmaxOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph);

  virtual void setup() override final;
};
} // namespace neuralnet

#endif
