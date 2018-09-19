#include <neuralnet/relu.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

ReluOp::ReluOp(const onnx::NodeProto &node, Graph *pgraph) : Op(node, pgraph) {

  // //std::cout << "in relu constructor" << std::endl;
}

void ReluOp::setup() {

  // //std::cout << "in relu setup" << std::endl;
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace neuralnet
