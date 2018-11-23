#include <poponnx/op/squeeze.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

SqueezeOp::SqueezeOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::vector<std::unique_ptr<Op>> SqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new SqueezeGradOp(this)));
  return upops;
}

std::unique_ptr<Op> SqueezeOp::clone() const {
  return std::unique_ptr<Op>(new SqueezeOp(*this));
}

void SqueezeOp::setup() {
  output.tensor(0)->info = {input.tensor(0)->info.dataType(),
                            squeeze(input.tensor(0)->info.shape())};
}

void SqueezeGradOp::setup() { output.tensor(0)->info = unsqueezedInfo; }

SqueezeGradOp::SqueezeGradOp(SqueezeOp *op_)
    : Op({"SqueezeGrad", op_->pir, {}, getPoponnxDomain()}),
      unsqueezedInfo(op_->input.tensor(0)->info) {}

const std::vector<GradInOutMapper> &SqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of squeeze
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &SqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

} // namespace willow
