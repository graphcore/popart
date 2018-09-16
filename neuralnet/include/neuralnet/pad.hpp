#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class PadOp : public NonGradOp {
public:
  PadOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : NonGradOp(opId, node, pgraph) {}
};
} // namespace neuralnet

#endif
