#include <poponnx/squeeze.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

SqueezeOp::SqueezeOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {}

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
// input.tensor(0)->info; }

void SqueezeGradOp::setup() { output.tensor(0)->info = unsqueezedInfo; }

SqueezeGradOp::SqueezeGradOp(SqueezeOp *op_)
    : Op({"SqueezeGrad", op_->pir, {}, getWillowDomain()}),
      unsqueezedInfo(op_->input.tensor(0)->info) {}

const std::vector<GradInOutMapper> &SqueezeGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createSqueezeGradInfo();
  return inInfo;
}

std::map<int, int> SqueezeGradOp::createSqueezeGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  return {{0, 0}};
}

const std::map<int, int> &SqueezeGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createSqueezeGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper> SqueezeGradOp::createSqueezeGradInfo() const {
  // input at index 0 : gradient of output of squeeze
  return {{0, 0, GradOpInType::GRADOUT}};
}

} // namespace willow
