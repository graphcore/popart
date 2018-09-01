#ifndef GUARD_NEURALNET_AVERAGEPOOL_HPP
#define GUARD_NEURALNET_AVERAGEPOOL_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet{

class AveragePoolOp: public Op{
public:
  AveragePoolOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : Op(opId, node, pgraph) {}

};
}

#endif
