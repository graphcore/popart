#include <memory>
#include <popart/op/identity.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

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

std::unique_ptr<Op>
IdentityOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::IdentityInplace) {
    return std::make_unique<IdentityInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}
std::vector<std::tuple<OperatorIdentifier, float>>
IdentityOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::IdentityInplace, 10}};
}

IdentityInplaceOp::IdentityInplaceOp(const IdentityOp &op)
    : IdentityOp(Onnx::CustomOperators::IdentityInplace, op.settings) {}

std::unique_ptr<Op> IdentityInplaceOp::clone() const {
  return std::make_unique<IdentityInplaceOp>(*this);
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

} // namespace popart
