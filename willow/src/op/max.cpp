#include <poponnx/makeunique.hpp>
#include <poponnx/op/max.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MaxOp::MaxOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MaxOp::clone() const { return make_unique<MaxOp>(*this); }

std::unique_ptr<Op> MaxOp::getIthGrad(int i) const {
  return make_unique<MaxArgGradOp>(*this, i);
}

MaxArgGradOp::MaxArgGradOp(const MaxOp &op_, InIndex inputIndex)
    : NonLinearVariadicGradOp(Onnx::GradOperators::MaxArgGrad,
                              op_,
                              inputIndex) {}

std::unique_ptr<Op> MaxArgGradOp::clone() const {
  return make_unique<MaxArgGradOp>(*this);
}

namespace {
static OpCreator<MaxOp> maxOpCreator({Onnx::Operators::Max_6,
                                      Onnx::Operators::Max_8});
} // namespace

} // namespace poponnx
