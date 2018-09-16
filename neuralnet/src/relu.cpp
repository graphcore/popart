#include <neuralnet/relu.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

ReluOp::ReluOp(OpId opId, const onnx::NodeProto &node, Graph *pgraph)
    : NonGradOp(opId, node, pgraph) {

  // //std::cout << "in relu constructor" << std::endl;
}

void ReluOp::setup() {

  // //std::cout << "in relu setup" << std::endl;
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace neuralnet
