#include <poponnx/makeunique.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SumOp::SumOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> SumOp::clone() const { return make_unique<SumOp>(*this); }

// The output info is the same as the first input info
// Assumption : there is more at least input tensor
// TODO T5688 numpy style broadcasting.
void SumOp::setup() { outInfo(getOutIndex()) = inInfo(0); }

namespace {
static OpCreator<SumOp> sumOpCreator(Onnx::Operators::Sum);
}

} // namespace poponnx
