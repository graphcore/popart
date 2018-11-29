#include <poponnx/error.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

SumOp::SumOp(const OpConstructorBundle &bundle) : Op(bundle) {}

SumOp::SumOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

std::unique_ptr<Op> SumOp::clone() const {
  return std::unique_ptr<Op>(new SumOp(*this));
}

// TODO T5688 numpy style broadcasting.
void SumOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

} // namespace willow
