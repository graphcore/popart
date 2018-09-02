#ifndef GUARD_NEURALNET_RELU_HPP
#define GUARD_NEURALNET_RELU_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class ReluOp : public Op {
public:
  ReluOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : Op(opId, node, pgraph) {}
};
} // namespace neuralnet

#endif
