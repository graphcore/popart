#include <poponnx/negate.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

NegateOp::NegateOp(const OpConstructorBundle &bundle) : Op(bundle) {}

NegateOp::NegateOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {}

std::unique_ptr<Op> NegateOp::clone() const {
  return std::unique_ptr<Op>(new NegateOp(*this));
}

std::vector<std::unique_ptr<Op>> NegateOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(new NegateGradOp(this));
  return upops;
}

void NegateOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

NegateGradOp::NegateGradOp(NegateOp *fwdOp)
    : NegateOp({"NegateGrad", fwdOp->pir, {}, getWillowDomain()}) {}

std::unique_ptr<Op> NegateGradOp::clone() const {
  return std::unique_ptr<Op>(new NegateGradOp(*this));
}

const std::vector<GradInOutMapper> &NegateGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &NegateGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};

  return outInfo;
}

} // namespace willow
