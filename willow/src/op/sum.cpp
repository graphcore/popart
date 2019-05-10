#include <poponnx/makeunique.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

SumOp::SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SumOp::clone() const { return make_unique<SumOp>(*this); }

std::unique_ptr<Op> SumOp::getIthGrad(int i) const {
  return make_unique<SumArgGradOp>(*this, i);
}

SumArgGradOp::SumArgGradOp(const SumOp &op_, InIndex inputIndex)
    : LinearVariadicGradOp(Onnx::GradOperators::SumArgGrad, op_, inputIndex) {

  gradInputInfoVec = {
      {getGradInIndex(), VariadicOp::getOutIndex(), GradOpInType::GRADOUT}};
}

std::unique_ptr<Op> SumArgGradOp::clone() const {
  return make_unique<SumArgGradOp>(*this);
}

const std::vector<GradInOutMapper> &SumArgGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

namespace {
static OpCreator<SumOp> sumOpCreator({Onnx::Operators::Sum_6,
                                      Onnx::Operators::Sum_8});
} // namespace

} // namespace poponnx
