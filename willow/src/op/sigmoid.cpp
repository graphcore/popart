#include <poponnx/makeunique.hpp>
#include <poponnx/op/sigmoid.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SigmoidOp::SigmoidOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SigmoidOp::clone() const {
  return make_unique<SigmoidOp>(*this);
}

std::vector<std::unique_ptr<Op>> SigmoidOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SigmoidGradOp>(*this));
  return upops;
}

SigmoidGradOp::SigmoidGradOp(const SigmoidOp &fwdOp)
    : Op(Onnx::GradOperators::SigmoidGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> SigmoidGradOp::clone() const {
  return make_unique<SigmoidGradOp>(*this);
}

const std::vector<GradInOutMapper> &SigmoidGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), SigmoidOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(), SigmoidOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &SigmoidGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SigmoidOp::getInIndex()}};

  return outInfo;
}

void SigmoidGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getFwdOutInIndex());
}

namespace {
static OpCreator<SigmoidOp> sigmoidOpCreator(Onnx::Operators::Sigmoid_6);
} // namespace

} // namespace poponnx
