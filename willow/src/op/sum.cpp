#include <poponnx/makeunique.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SumOp::SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SumOp::clone() const { return make_unique<SumOp>(*this); }

// The output info is the same as the first input info
// Assumption : there is more at least input tensor
// TODO T5688 numpy style broadcasting.
void SumOp::setup() { outInfo(getOutIndex()) = inInfo(0); }

namespace {
static OpCreator<SumOp> sumOpCreator({Onnx::Operators::Sum_6,
                                      Onnx::Operators::Sum_8});
} // namespace

} // namespace poponnx
