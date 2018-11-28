#include <poponnx/op/mul.hpp>
#include <poponnx/tensor.hpp>

namespace willow {

InIndex MulOp::arg0Index() { return 0; }
InIndex MulOp::arg1Index() { return 1; }

MulOp::MulOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

MulOp::MulOp(const OpConstructorBundle &bundle) : Op(bundle) {}

std::unique_ptr<Op> MulOp::clone() const {
  return std::unique_ptr<Op>(new MulOp(*this));
}

std::vector<std::unique_ptr<Op>> MulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_in_0   = input.tensor(arg0Index())->info.shape();
  const auto &shape_in_1   = input.tensor(arg1Index())->info.shape();
  const auto &shape_output = output.tensor(0)->info.shape();

  upops.emplace_back(std::unique_ptr<Op>(
      new MulArg0GradOp(this, npReductionAxis(shape_in_0, shape_output))));
  upops.emplace_back(std::unique_ptr<Op>(
      new MulArg1GradOp(this, npReductionAxis(shape_in_1, shape_output))));
  return upops;
}

void MulOp::setup() {
  output.tensor(0)->info = npOut(input.tensor(0)->info, input.tensor(1)->info);
}

MulArgGradOp::MulArgGradOp(const OpConstructorBundle &bundle,
                           const std::vector<int64_t> &reduction_axes_,
                           const TensorInfo &forward_op_arg_info_)
    : Op(bundle), reduction_axes(reduction_axes_),
      forward_op_arg_info(forward_op_arg_info_) {}

const std::vector<int64_t> &MulArgGradOp::getReductionAxes() {
  return reduction_axes;
}

void MulArgGradOp::setup() { output.tensor(0)->info = forward_op_arg_info; }

MulArg0GradOp::MulArg0GradOp(MulOp *op_,
                             const std::vector<int64_t> &reduction_axes)
    : MulArgGradOp({"MulArg0Grad", op_->pir, {}, getPoponnxDomain()},
                   reduction_axes,
                   op_->input.tensor(MulOp::arg0Index())->info) {}

const std::map<int, int> &MulArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, MulOp::arg0Index()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &MulArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}, {1, MulOp::arg1Index(), GradOpInType::IN}};
  return inInfo;
}

MulArg1GradOp::MulArg1GradOp(MulOp *op_,
                             const std::vector<int64_t> &reduction_axes)
    : MulArgGradOp({"MulArg1Grad", op_->pir, {}, getPoponnxDomain()},
                   reduction_axes,
                   op_->input.tensor(MulOp::arg1Index())->info) {}

const std::map<int, int> &MulArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, MulOp::arg1Index()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &MulArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, 0, GradOpInType::GRADOUT}, {1, MulOp::arg0Index(), GradOpInType::IN}};
  return inInfo;
}

} // namespace willow
