#include <poponnx/makeunique.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SquareOp::SquareOp(const OperatorIdentifier &_opid,
                   Ir *_ir,
                   const std::string &name,
                   const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> SquareOp::clone() const {
  return std::unique_ptr<Op>(new SquareOp(*this));
}

std::vector<std::unique_ptr<Op>> SquareOp::getGradOps() {
  throw error("Grad op has not been implemented for SquareOp");
}

namespace {
static OpCreator<SquareOp> squareOpCreator(Onnx::CustomOperators::Square);
}

} // namespace poponnx
