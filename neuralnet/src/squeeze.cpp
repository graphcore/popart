#include <neuralnet/squeeze.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

SqueezeOp::SqueezeOp(const onnx::NodeProto &node, Graph *pgraph)
    : Op(node, pgraph) {}

std::vector<std::unique_ptr<Op>> SqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::unique_ptr<Op>(new SqueezeGradOp(this)));
  return upops;
}


std::unique_ptr<Op> SqueezeOp::clone() const {
  return std::unique_ptr<Op>( new SqueezeOp(*this));
}

void SqueezeOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

void SqueezeGradOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

SqueezeGradOp::SqueezeGradOp(SqueezeOp *op_)
    : GradOp({"SqueezeGrad", op_->pgraph, {}, getNeuralNetDomain()}),
      squeezeOp(op_) {}

Op *SqueezeGradOp::getNonGradCreator() const { return squeezeOp; }

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

} // namespace neuralnet
