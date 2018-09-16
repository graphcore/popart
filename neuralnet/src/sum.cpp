#include <neuralnet/error.hpp>
#include <neuralnet/sum.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

SumOp::SumOp(const OpConstructorBundle &bundle) : NonGradOp(bundle) {}

void SumOp::setup() {
  throw error("SumOp setup TODO");
  // output.setInfoIfIndex(input.tensor(0)->info, 0);
}

} // namespace neuralnet
