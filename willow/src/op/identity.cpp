#include <poponnx/makeunique.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

IdentityOp::IdentityOp(const OpConstructorBundle &bundle) : Op(bundle) {}

IdentityOp::IdentityOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir) {}

std::unique_ptr<Op> IdentityOp::clone() const {
  return make_unique<IdentityOp>(*this);
}

std::vector<std::unique_ptr<Op>> IdentityOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<IdentityGradOp>(this));
  return upops;
}

void IdentityOp::setup() { outInfo(0) = inInfo(0); }

IdentityGradOp::IdentityGradOp(IdentityOp *fwdOp)
    : IdentityOp({"IdentityGrad", fwdOp->pir, {}, getPoponnxDomain()}) {}

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

} // namespace poponnx
