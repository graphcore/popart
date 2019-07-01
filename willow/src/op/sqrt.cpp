#include <memory>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SqrtOp::SqrtOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SqrtOp::clone() const {
  return std::make_unique<SqrtOp>(*this);
}

std::vector<std::unique_ptr<Op>> SqrtOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SqrtGradOp>(*this));
  return upops;
}

SqrtGradOp::SqrtGradOp(const SqrtOp &fwdOp)
    : Op(Onnx::GradOperators::SqrtGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> SqrtGradOp::clone() const {
  return std::make_unique<SqrtGradOp>(*this);
}

void SqrtGradOp::setup() { outInfo(getOutIndex()) = inInfo(getGradInIndex()); }

const std::vector<GradInOutMapper> &SqrtGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), 0, GradOpInType::GRADOUT},
      {getFwdOutInIndex(), SqrtOp::getOutIndex(), GradOpInType::OUT}};

  return inInfo;
}

const std::map<int, int> &SqrtGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqrtOp::getInIndex()}};

  return outInfo;
}

namespace {
static OpCreator<SqrtOp> sqrtOpCreator(Onnx::Operators::Sqrt_6);
} // namespace

} // namespace poponnx
