#ifndef GUARD_NEURALNET_LOGSOFTMAX_HPP
#define GUARD_NEURALNET_LOGSOFTMAX_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {
class LogSoftmaxOp : public Op {
public:
  LogSoftmaxOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : Op(opId, node, pgraph) {}
};
} // namespace neuralnet

#endif
