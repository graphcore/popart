#include <poponnx/makeunique.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SquareOp::SquareOp(const OpConstructorBundle &bundle)
    : ElementWiseUnaryOp(bundle) {}

SquareOp::SquareOp(const onnx::NodeProto &node, Ir *ir)
    : ElementWiseUnaryOp(node, ir) {}

std::unique_ptr<Op> SquareOp::clone() const {
  return std::unique_ptr<Op>(new SquareOp(*this));
}

std::vector<std::unique_ptr<Op>> SquareOp::getGradOps() {
  throw error("Grad op has not been implemented for SquareOp");
}

} // namespace poponnx
