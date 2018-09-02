#ifndef GUARD_NEURALNET_CONV_HPP
#define GUARD_NEURALNET_CONV_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <cblas.h>
#pragma clang diagnostic pop // stop ignoring warnings

#include <neuralnet/graph.hpp>

namespace neuralnet {

class ConvOp : public Op {
public:
  ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
      : Op(opId, node, pgraph) {}
};
} // namespace neuralnet

#endif
