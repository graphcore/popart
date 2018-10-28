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
    : GradOp({"SoftmaxGrad", op_->pir, {}, getWillowDomain()}),
      logsoftmaxOp(op_) {}

Op *SoftmaxGradOp::getNonGradCreator() const { return logsoftmaxOp; }

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

std::vector<GradInOutMapper> SoftmaxGradOp::createSoftmaxGradInfo() const {
  // input at index 0 : gradient of output of logsoftmax
  // input at index 1 : output of logsoftmax (p's)
  // the (1-sparse) gradient of the output will be used to determine
  // which index gets 1 - p, instead of - p .
  return {{0, 0, GradOpInType::GRADOUT}, {1, 0, GradOpInType::OUT}};
}

SoftmaxGradDirectOp::SoftmaxGradDirectOp(Op *op)
    : Op({"SoftmaxGradDirect", // op_type
          op->pir,             //
          {},                  // no Attributes
          getWillowDomain()}) {
  if (op->opType != OpType::SOFTMAX) {
    throw error("Require SoftmaxOp in SoftmaxGradDirectOp constructor, not " +
                op->op_type());
  }
  logsoftmaxOp = static_cast<SoftmaxOp *>(op);
}

std::vector<std::unique_ptr<Op>> SoftmaxGradDirectOp::getGradOps() {
  throw error("SoftmaxGradDirectOp is not a true non-grad op, no getGradOps");
}

SoftmaxOp *SoftmaxGradDirectOp::getSoftmaxOp() const { return logsoftmaxOp; }

std::unique_ptr<Op> SoftmaxGradDirectOp::clone() const {
  throw error("Unexpected (but valid) request to clone SoftmaxGradDirectOp");
}

void SoftmaxGradDirectOp::setup() {
  // gradient of activations has same shape as probabilities
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace willow
