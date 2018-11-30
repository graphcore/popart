#include <poponnx/makeunique.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SquareOp::SquareOp(const OpConstructorBundle &bundle) : Op(bundle) {}

SquareOp::SquareOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

std::unique_ptr<Op> SquareOp::clone() const {
  return std::unique_ptr<Op>(new SquareOp(*this));
}

std::vector<std::unique_ptr<Op>> SquareOp::getGradOps() {
  throw error("Grad op has not been implemented for SquareOp");
}

void SquareOp::setup() { outInfo(0) = inInfo(0); }

} // namespace poponnx
