#include <poponnx/makeunique.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

IdentityOp::IdentityOp(const OperatorIdentifier &_opid,
                       Ir *_ir,
                       const std::string &name,
                       const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> IdentityOp::clone() const {
  return make_unique<IdentityOp>(*this);
}

std::vector<std::unique_ptr<Op>> IdentityOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<IdentityGradOp>(this));
  return upops;
}

IdentityGradOp::IdentityGradOp(IdentityOp *fwdOp)
    : IdentityOp(Onnx::GradOperators::IdentityGrad, fwdOp->pir) {}

std::unique_ptr<Op> IdentityGradOp::clone() const {
  return make_unique<IdentityGradOp>(*this);
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
static OpCreator<IdentityOp> identityOpCreator(Onnx::Operators::Identity);
static GradOpCreator<IdentityGradOp>
    identityGradOpCreator(Onnx::GradOperators::IdentityGrad);
} // namespace

} // namespace poponnx
