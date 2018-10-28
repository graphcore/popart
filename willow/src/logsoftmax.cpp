#include <willow/error.hpp>
#include <willow/logsoftmax.hpp>
#include <willow/tensor.hpp>

namespace willow {

LogSoftmaxOp::LogSoftmaxOp(const onnx::NodeProto &node, Ir *pir)
    : Op(node, pir) {}

void LogSoftmaxOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new LogSoftmaxGradOp(this)));
  return upops;
}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return std::unique_ptr<Op>(new LogSoftmaxOp(*this));
}

void LogSoftmaxGradOp::setup() {
  output.tensor(0)->info = input.tensor(0)->info;
}

LogSoftmaxGradOp::LogSoftmaxGradOp(LogSoftmaxOp *op_)
    : GradOp({"LogSoftmaxGrad", op_->pir, {}, getWillowDomain()}),
      logsoftmaxOp(op_) {}

Op *LogSoftmaxGradOp::getNonGradCreator() const { return logsoftmaxOp; }

const std::vector<GradInOutMapper> &LogSoftmaxGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = createLogSoftmaxGradInfo();
  return inInfo;
}

std::map<int, int> LogSoftmaxGradOp::createLogSoftmaxGradOutToIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  return {{0, 0}};
}

const std::map<int, int> &LogSoftmaxGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = createLogSoftmaxGradOutToIn();
  return outInfo;
}

std::vector<GradInOutMapper>
LogSoftmaxGradOp::createLogSoftmaxGradInfo() const {
  // input at index 0 : gradient of output of logsoftmax
  // input at index 1 : output of logsoftmax (p's)
  // the (1-sparse) gradient of the output will be used to determine
  // which index gets 1 - p, instead of - p .
  return {{0, 0, GradOpInType::GRADOUT}, {1, 0, GradOpInType::OUT}};
}

LogSoftmaxGradDirectOp::LogSoftmaxGradDirectOp(Op *op)
    : Op({"LogSoftmaxGradDirect", // op_type
          op->pir,                //
          {},                     // no Attributes
          getWillowDomain()}) {
  if (op->opType != OpType::LOGSOFTMAX) {
    throw error(
        "Require LogSoftmaxOp in LogSoftmaxGradDirectOp constructor, not " +
        op->op_type());
  }
  logsoftmaxOp = static_cast<LogSoftmaxOp *>(op);
}

std::vector<std::unique_ptr<Op>> LogSoftmaxGradDirectOp::getGradOps() {
  throw error(
      "LogSoftmaxGradDirectOp is not a true non-grad op, no getGradOps");
}

LogSoftmaxOp *LogSoftmaxGradDirectOp::getLogSofmaxOp() const {
  return logsoftmaxOp;
}

std::unique_ptr<Op> LogSoftmaxGradDirectOp::clone() const {
  throw error("Unexpected (but valid) request to clone LogSoftmaxGradDirectOp");
}

void LogSoftmaxGradDirectOp::setup() {
  // gradient of activations has same shape as probabilities
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace willow
