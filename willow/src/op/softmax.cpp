#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SoftmaxOp::SoftmaxOp(const onnx::NodeProto &node, Ir *_pir)
    : ElementWiseUnaryOp(node, _pir) {}

std::vector<std::unique_ptr<Op>> SoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SoftmaxGradOp>(this));
  return upops;
}

std::unique_ptr<Op> SoftmaxOp::clone() const {
  return make_unique<SoftmaxOp>(*this);
}

void SoftmaxGradOp::setup() {
  outInfo(getOutIndex()) = inInfo(getGradProbsInIndex());
}

SoftmaxGradOp::SoftmaxGradOp(SoftmaxOp *op_)
    : Op({OpType::SOFTMAXGRAD, op_->pir, {}}) {}

const std::vector<GradInOutMapper> &SoftmaxGradOp::gradInputInfo() const {
  // input at index 0 (probGradInputIndex()) : gradient of output of softmax
  // input at index 1 (actsIn()): input of softmax (activations before p)
  // the (1-sparse) gradient of the output will be used
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradProbsInIndex(), SoftmaxOp::getOutIndex(), GradOpInType::GRADOUT},
      {getActsInIndex(), SoftmaxOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::map<int, int> &SoftmaxGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SoftmaxOp::getInIndex()}};
  return outInfo;
}

SoftmaxGradDirectOp::SoftmaxGradDirectOp(Ir *ir, const NllLoss *nls)
    : Op({OpType::SOFTMAXGRADDIRECT, ir, {}}) {
  nllloss_ = nls;
}

const NllLoss *SoftmaxGradDirectOp::nlll() const { return nllloss_; }

std::vector<std::unique_ptr<Op>> SoftmaxGradDirectOp::getGradOps() {
  throw error("SoftmaxGradDirectOp is not a true non-grad op, no getGradOps");
}

std::unique_ptr<Op> SoftmaxGradDirectOp::clone() const {
  throw error("Unexpected (but valid) request to clone SoftmaxGradDirectOp");
}

void SoftmaxGradDirectOp::setup() {
  // gradient of activations has same shape as probabilities
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

} // namespace poponnx
