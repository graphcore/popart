#include <poponnx/makeunique.hpp>
#include <poponnx/op/sign.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

SignOp::SignOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> SignOp::clone() const { return make_unique<SignOp>(*this); }

std::vector<std::unique_ptr<Op>> SignOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SignGradOp>(*this));
  return upops;
}

void SignOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

OperatorIdentifier SignOp::getOpId(const Ir &) {
  return Onnx::Operators::Sign_9;
}

SignGradOp::SignGradOp(const SignOp &op_)
    : Op(Onnx::GradOperators::SignGrad, op_.getSettings()) {}

const std::map<int, int> &SignGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SignOp::getInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &SignGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SignOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void SignGradOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

namespace {
static OpCreator<SignOp> absOpCreator({Onnx::Operators::Sign_9});
} // namespace

} // namespace poponnx
