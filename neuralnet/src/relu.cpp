#include <neuralnet/relu.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {


void ReluOp::inferInfo() {
  output.tensor(0)->info = input.tensor(0)->info;
}

}
