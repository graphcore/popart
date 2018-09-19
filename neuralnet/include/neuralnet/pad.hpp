#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class PadOp : public Op {
public:
  PadOp(const onnx::NodeProto &node, Graph *pgraph) : Op(node, pgraph) {}
};
} // namespace neuralnet

#endif
