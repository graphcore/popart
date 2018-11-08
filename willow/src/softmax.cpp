#include <willow/error.hpp>
#include <willow/softmax.hpp>
#include <willow/tensor.hpp>

namespace willow {

SoftmaxOp::SoftmaxOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {}

void SoftmaxOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

std::vector<std::unique_ptr<Op>> SoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new SoftmaxGradOp(this)));
  return upops;
}

std::unique_ptr<Op> SoftmaxOp::clone() const {
  return std::unique_ptr<Op>(new SoftmaxOp(*this));
}

void SoftmaxGradOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

SoftmaxGradOp::SoftmaxGradOp(SoftmaxOp *op_)
    : Op({"SoftmaxGrad", op_->pir, {}, getWillowDomain()}) {}

const std::vector<GradInOutMapper> &SoftmaxGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createSoftmaxGradInfo();
  return inInfo;
}

std::map<int, int> SoftmaxGradOp::createSoftmaxGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  return {{0, 0}};
}

const std::map<int, int> &SoftmaxGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createSoftmaxGradOutToIn();
  return outInfo;
}

int SoftmaxGradOp::gradProbsIn() const { return 0; }

int SoftmaxGradOp::actsIn() const { return 1; }

std::vector<GradInOutMapper> SoftmaxGradOp::createSoftmaxGradInfo() const {
  // input at index 0 (probGradInputIndex()) : gradient of output of softmax
  // input at index 1 (actsIn()): input of softmax (activations before p)
  // the (1-sparse) gradient of the output will be used
  return {{gradProbsIn(), 0, GradOpInType::GRADOUT},
          {actsIn(), 0, GradOpInType::IN}};
}

SoftmaxGradDirectOp::SoftmaxGradDirectOp(Ir *ir,
                                         const NllLoss *nls)
    : Op({"SoftmaxGradDirect", // op_type
          ir,                  //
          {},                  // no Attributes
          getWillowDomain()}) {
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
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace willow
