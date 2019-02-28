#include <poponnx/makeunique.hpp>
#include <poponnx/op/mean.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

MeanOp::MeanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : VariadicOp(_opid, settings_) {}

std::unique_ptr<Op> MeanOp::clone() const { return make_unique<MeanOp>(*this); }

std::unique_ptr<Op> MeanOp::getIthGrad(int i) const {
  return std::unique_ptr<MeanArgGradOp>(new MeanArgGradOp(*this, i));
}

MeanArgGradOp::MeanArgGradOp(const MeanOp &op_, InIndex inputIndex)
    : LinearVariadicGradOp(Onnx::GradOperators::MeanArgGrad, op_, inputIndex) {

  gradInputInfoVec = {
      {getGradInIndex(), VariadicOp::getOutIndex(), GradOpInType::GRADOUT}};

  nInputs = op_.input->n();
}

const std::vector<GradInOutMapper> &MeanArgGradOp::gradInputInfo() const {
  return gradInputInfoVec;
}

namespace {
static OpCreator<MeanOp> opCreator({Onnx::Operators::Mean_6,
                                    Onnx::Operators::Mean_8});
} // namespace

} // namespace poponnx
