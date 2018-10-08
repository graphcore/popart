#include <willow/error.hpp>
#include <willow/sum.hpp>
#include <willow/tensor.hpp>

namespace willow {

SumOp::SumOp(const OpConstructorBundle &bundle) : Op(bundle) {}

std::unique_ptr<Op> SumOp::clone() const {
  return std::unique_ptr<Op>(new SumOp(*this));
}

void SumOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

} // namespace willow
