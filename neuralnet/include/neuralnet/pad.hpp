#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet{

class PadOp: public Op{
public:
  PadOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : Op(opId, node, pgraph) {}

};
}

#endif
