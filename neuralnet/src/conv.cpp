#include <neuralnet/conv.hpp>
#include <neuralnet/error.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

ConvOp::ConvOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
    : Op(opId, node, pgraph) {

  int nInputs = 0;
  for (auto &id : node.input()) {
    if (id != "") {
      ++nInputs;
    }
  }

  // handling this CONV as a special case, as it
  // needs splitting into 2 (CONV and add bias)
  // will still add, changed in optimise step.
  // this first step builds exactly 1-1 with onnx graph
  // and then later do the split
  if (nInputs == 3) {
    throw error("Conv with bias case not handled");
  }
}
} // namespace neuralnet
