#include <poponnx/makeunique.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

DivOp::DivOp(const onnx::NodeProto &node, Ir *ir) : Op(node, ir) {}

DivOp::DivOp(const OpConstructorBundle &bundle) : Op(bundle) {}

std::unique_ptr<Op> DivOp::clone() const { return make_unique<DivOp>(*this); }

std::vector<std::unique_ptr<Op>> DivOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_in_0   = inShape(getArg0InIndex());
  const auto &shape_in_1   = inShape(getArg1InIndex());
  const auto &shape_output = outShape(getOutIndex());

  upops.emplace_back(make_unique<DivArg0GradOp>(
      this, npReductionAxis(shape_in_0, shape_output)));
  upops.emplace_back(make_unique<DivArg1GradOp>(
      this, npReductionAxis(shape_in_1, shape_output)));
  return upops;
}

void DivOp::setup() {
  outInfo(getOutIndex()) =
      npOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

DivArgGradOp::DivArgGradOp(const OpConstructorBundle &bundle,
                           const std::vector<int64_t> &reduction_axes_,
                           const TensorInfo &forward_op_arg_info_)
    : Op(bundle), forward_op_arg_info(forward_op_arg_info_),
      reduction_axes(reduction_axes_) {}

void DivArgGradOp::setup() { outInfo(0) = forward_op_arg_info; }

const std::vector<int64_t> &DivArgGradOp::getReductionAxes() const {
  return reduction_axes;
}

DivArg0GradOp::DivArg0GradOp(DivOp *op,
                             const std::vector<int64_t> &reduction_axes_)
    : DivArgGradOp({"DivArg0Grad", op->pir, {}, getPoponnxDomain()},
                   reduction_axes_,
                   op->inInfo(DivOp::getArg0InIndex())) {}

const std::map<int, int> &DivArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DivOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &DivArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, DivOp::getArg1InIndex(), GradOpInType::IN}};
  return inInfo;
}

DivArg1GradOp::DivArg1GradOp(DivOp *op,
                             const std::vector<int64_t> &reduction_axes_)
    : DivArgGradOp({"DivArg1Grad", op->pir, {}, getPoponnxDomain()},
                   reduction_axes_,
                   op->inInfo(DivOp::getArg1InIndex())) {}

const std::map<int, int> &DivArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), DivOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &DivArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, DivOp::getArg0InIndex(), GradOpInType::IN},
      {2, DivOp::getArg1InIndex(), GradOpInType::IN}};
  return inInfo;
}

} // namespace poponnx
