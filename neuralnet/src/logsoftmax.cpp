#include <neuralnet/logsoftmax.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

LogSoftmaxOp::LogSoftmaxOp(const onnx::NodeProto &node, Graph *pgraph)
    : Op(node, pgraph) {}

void LogSoftmaxOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new LogSoftmaxGradOp(this)));
  return upops;
}

void LogSoftmaxGradOp::setup() {
  output.tensor(0)->info = input.tensor(0)->info;
}

LogSoftmaxGradOp::LogSoftmaxGradOp(LogSoftmaxOp *op_)
    : GradOp({"LogSoftmaxGrad", op_->pgraph, {}, getNeuralNetDomain()}),
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

} // namespace neuralnet
