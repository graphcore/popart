#include <poponnx/makeunique.hpp>
#include <poponnx/op/squeeze.hpp>

namespace poponnx {

SqueezeOp::SqueezeOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::vector<std::unique_ptr<Op>> SqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SqueezeGradOp>(this));
  return upops;
}

std::unique_ptr<Op> SqueezeOp::clone() const {
  return make_unique<SqueezeOp>(*this);
}

void SqueezeOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            squeeze(inShape(getInIndex()))};
}

void SqueezeGradOp::setup() { outInfo(getOutIndex()) = unsqueezedInfo; }

SqueezeGradOp::SqueezeGradOp(SqueezeOp *op_)
    : Op({OpType::SQUEEZEGRAD, op_->pir, {}}),
      unsqueezedInfo(op_->inInfo(SqueezeOp::getInIndex())) {}

const std::vector<GradInOutMapper> &SqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of squeeze
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SqueezeOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &SqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqueezeOp::getInIndex()}};
  return outInfo;
}

} // namespace poponnx
