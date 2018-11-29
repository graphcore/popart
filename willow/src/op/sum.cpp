#include <poponnx/error.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

SumOp::SumOp(const OpConstructorBundle &bundle) : Op(bundle) {}

SumOp::SumOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

std::unique_ptr<Op> SumOp::clone() const { return make_unique<SumOp>(*this); }

// The output info is the same as the first input info
// Assumption : there is more that 1 input tensor
void SumOp::setup() { outInfo(getOutIndex()) = inInfo(0); }

} // namespace poponnx
