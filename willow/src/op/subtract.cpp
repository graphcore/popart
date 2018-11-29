#include <poponnx/op/subtract.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

int SubtractOp::arg0Index() { return 0; }
int SubtractOp::arg1Index() { return 1; }

SubtractOp::SubtractOp(const onnx::NodeProto &node, Ir *_pir)
    : Op(node, _pir) {}

std::unique_ptr<Op> SubtractOp::clone() const {
  return std::unique_ptr<Op>(new SubtractOp(*this));
}

std::vector<std::unique_ptr<Op>> SubtractOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = input.tensor(SubtractOp::arg0Index())->info.shape();
  const auto &shape_o0 = output.tensor(0)->info.shape();

  upops.emplace_back(std::unique_ptr<Op>(
      new SubtractArg0GradOp(this, npReductionAxis(shape_a0, shape_o0))));
  upops.emplace_back(std::unique_ptr<Op>(new SubtractArg1GradOp(this)));
  return upops;
}

void SubtractOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

SubtractArg0GradOp::SubtractArg0GradOp(SubtractOp *op_,
                                       const std::vector<int64_t> &_axes)
    : ReduceSumOp({"SubtractArg0Grad", op_->pir, {}, getPoponnxDomain()},
                  _axes,
                  false),
      forward_op_arg_info(op_->input.tensor(SubtractOp::arg0Index())->info) {}

const std::map<int, int> &SubtractArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, SubtractOp::arg0Index()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

void SubtractArg0GradOp::setup() {
  output.tensor(0)->info = forward_op_arg_info;
}

SubtractArg1GradOp::SubtractArg1GradOp(SubtractOp *op_)
    : NegateOp({"SubtractArg1Grad", op_->pir, {}, getPoponnxDomain()}),
      forward_op_arg_info(op_->input.tensor(SubtractOp::arg1Index())->info) {}

const std::map<int, int> &SubtractArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, SubtractOp::arg1Index()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}};

  return inInfo;
}

std::unique_ptr<Op> SubtractArg1GradOp::clone() const {
  return make_unique<SubtractArg1GradOp>(*this);
}

void SubtractArg1GradOp::setup() {
  output.tensor(0)->info = forward_op_arg_info;
}

} // namespace poponnx
