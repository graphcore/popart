#include <neuralnet/error.hpp>
#include <neuralnet/sum.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

SumOp::SumOp(const OpConstructorBundle &bundle) : Op(bundle) {}

void SumOp::setup() {
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace neuralnet
