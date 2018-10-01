#include <neuralnet/error.hpp>
#include <neuralnet/sum.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

SumOp::SumOp(const OpConstructorBundle &bundle) : Op(bundle) {}

std::unique_ptr<Op> SumOp::clone() const {
  return std::unique_ptr<Op>( new SumOp(*this));
}

void SumOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

} // namespace neuralnet
