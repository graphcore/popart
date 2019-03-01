#include <poponnx/makeunique.hpp>
#include <poponnx/op/min.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MinOp::MinOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MinOp::clone() const { return make_unique<MinOp>(*this); }

std::unique_ptr<Op> MinOp::getIthGrad(int i) const {
  return std::unique_ptr<MinArgGradOp>(new MinArgGradOp(*this, i));
}

MinArgGradOp::MinArgGradOp(const MinOp &op_, InIndex inputIndex)
    : NonLinearVariadicGradOp(Onnx::GradOperators::MinArgGrad,
                              op_,
                              inputIndex) {}

namespace {
static OpCreator<MinOp> minOpCreator({Onnx::Operators::Min_6,
                                      Onnx::Operators::Min_8});
} // namespace

} // namespace poponnx
