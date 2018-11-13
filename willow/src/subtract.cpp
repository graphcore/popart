#include <poponnx/subtract.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

SubtractOp::SubtractOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {}

std::unique_ptr<Op> SubtractOp::clone() const {
  return std::unique_ptr<Op>(new SubtractOp(*this));
}

std::vector<std::unique_ptr<Op>> SubtractOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new SubtractGradOp(this)));
  return upops;
}

void SubtractOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

void SubtractGradOp::setup() {
  // shapes and types of gradients are the same as the inputs
  output.tensor(0)->info = info0;
  output.tensor(1)->info = info1;
}

SubtractGradOp::SubtractGradOp(SubtractOp *op_)
    : Op({"SubtractGrad", op_->pir, {}, getWillowDomain()}),
      info0(op_->input.tensor(0)->info), info1(op_->input.tensor(1)->info) {}

const std::map<int, int> &SubtractGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  // ditto 1.
  static const std::map<int, int> outInfo = {{0, 0}, {1, 1}};
  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of subtract
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};
  return inInfo;
}

} // namespace willow
