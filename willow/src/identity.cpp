#include <poponnx/identity.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

IdentityOp::IdentityOp(const OpConstructorBundle &bundle) : Op(bundle) {}

IdentityOp::IdentityOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir) {}

std::unique_ptr<Op> IdentityOp::clone() const {
  return std::unique_ptr<Op>(new IdentityOp(*this));
}

std::vector<std::unique_ptr<Op>> IdentityOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(new IdentityGradOp(this));
  return upops;
}

void IdentityOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

IdentityGradOp::IdentityGradOp(IdentityOp *fwdOp)
    : IdentityOp({"IdentityGrad", fwdOp->pir, {}, getWillowDomain()}) {}

std::unique_ptr<Op> IdentityGradOp::clone() const {
  return std::unique_ptr<Op>(new IdentityGradOp(*this));
}

const std::vector<GradInOutMapper> &IdentityGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &IdentityGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};

  return outInfo;
}

} // namespace willow
