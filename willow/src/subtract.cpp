#include <poponnx/subtract.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

int SubtractOp::arg0Index() { return 0; }
int SubtractOp::arg1Index() { return 1; }

SubtractOp::SubtractOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir) {}

std::unique_ptr<Op> SubtractOp::clone() const {
  return std::unique_ptr<Op>(new SubtractOp(*this));
}

std::vector<std::unique_ptr<Op>> SubtractOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new SubtractArg0GradOp(this)));
  upops.emplace_back(std::unique_ptr<Op>(new SubtractArg1GradOp(this)));
  return upops;
}

void SubtractOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

SubtractArg0GradOp::SubtractArg0GradOp(SubtractOp *op_)
    : IdentityOp({"SubtractArg0Grad", op_->pir, {}, getWillowDomain()}) {}

const std::map<int, int> &SubtractArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, SubtractOp::arg0Index()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

SubtractArg1GradOp::SubtractArg1GradOp(SubtractOp *op_)
    : NegateOp({"SubtractArg1Grad", op_->pir, {}, getWillowDomain()}) {}

const std::map<int, int> &SubtractArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, SubtractOp::arg1Index()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

} // namespace willow
