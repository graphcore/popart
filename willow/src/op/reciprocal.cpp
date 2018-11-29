#include <poponnx/op/reciprocal.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReciprocalOp::ReciprocalOp(const OpConstructorBundle &bundle) : Op(bundle) {}

ReciprocalOp::ReciprocalOp(const onnx::NodeProto &node, Ir *ir)
    : Op(node, ir) {}

std::unique_ptr<Op> ReciprocalOp::clone() const {
  return std::unique_ptr<Op>(new ReciprocalOp(*this));
}

std::vector<std::unique_ptr<Op>> ReciprocalOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  upops.emplace_back(std::unique_ptr<Op>(new ReciprocalGradOp(this)));
  return upops;
}

void ReciprocalOp::setup() { output.tensor(0)->info = input.tensor(0)->info; }

ReciprocalGradOp::ReciprocalGradOp(ReciprocalOp *op_)
    : Op({"ReciprocalGrad", op_->pir, {}, getPoponnxDomain()}) {}

const std::map<int, int> &ReciprocalGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

const std::vector<GradInOutMapper> &ReciprocalGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};
  return inInfo;
}

void ReciprocalGradOp::setup() {
  output.tensor(0)->info = input.tensor(0)->info;
}

} // namespace poponnx
