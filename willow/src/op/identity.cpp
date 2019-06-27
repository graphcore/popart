#include <memory>
#include <poponnx/op/identity.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

IdentityOp::IdentityOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> IdentityOp::clone() const {
  return std::make_unique<IdentityOp>(*this);
}

std::vector<std::unique_ptr<Op>> IdentityOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<IdentityGradOp>(*this));
  return upops;
}

IdentityGradOp::IdentityGradOp(const IdentityOp &fwdOp)
    : IdentityOp(Onnx::GradOperators::IdentityGrad, fwdOp.getSettings()) {}

std::unique_ptr<Op> IdentityGradOp::clone() const {
  return std::make_unique<IdentityGradOp>(*this);
}

const std::vector<GradInOutMapper> &IdentityGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), IdentityOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &IdentityGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), IdentityOp::getInIndex()}};

  return outInfo;
}

namespace {
static OpCreator<IdentityOp> identityOpCreator(Onnx::Operators::Identity_1);
} // namespace

} // namespace poponnx
