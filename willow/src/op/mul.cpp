#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

MulOp::MulOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // TODO : Use the attributes in Mul-6
}

std::unique_ptr<Op> MulOp::clone() const { return make_unique<MulOp>(*this); }

std::vector<std::unique_ptr<Op>> MulOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_in_0   = inShape(getArg0InIndex());
  const auto &shape_in_1   = inShape(getArg1InIndex());
  const auto &shape_output = outShape(getOutIndex());

  upops.emplace_back(make_unique<MulArg0GradOp>(
      *this, npReductionAxis(shape_in_0, shape_output)));
  upops.emplace_back(make_unique<MulArg1GradOp>(
      *this, npReductionAxis(shape_in_1, shape_output)));
  return upops;
}

void MulOp::setup() {
  outInfo(getOutIndex()) =
      npOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

OperatorIdentifier MulOp::getOpId(const Ir &ir) {
  if (ir.getOpSetVersionFromModel(Domain::ai_onnx) >= 7) {
    return Onnx::Operators::Mul_7;
  } else {
    return Onnx::Operators::Mul_6;
  }
}

MulArgGradOp::MulArgGradOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &reduction_axes_,
                           const TensorInfo &forward_op_arg_info_,
                           const Op::Settings &settings_)
    : Op(_opid, settings_), reduction_axes(reduction_axes_),
      forward_op_arg_info(forward_op_arg_info_) {}

const std::vector<int64_t> &MulArgGradOp::getReductionAxes() {
  return reduction_axes;
}

void MulArgGradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

MulArg0GradOp::MulArg0GradOp(const MulOp &op_,
                             const std::vector<int64_t> &_reduction_axes)
    : MulArgGradOp(Onnx::GradOperators::MulArg0Grad,
                   _reduction_axes,
                   op_.inInfo(MulOp::getArg0InIndex()),
                   op_.getSettings()) {}

const std::map<int, int> &MulArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MulOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &MulArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, MulOp::getArg1InIndex(), GradOpInType::IN}};
  return inInfo;
}

MulArg1GradOp::MulArg1GradOp(const MulOp &op_,
                             const std::vector<int64_t> &_reduction_axes)
    : MulArgGradOp(Onnx::GradOperators::MulArg1Grad,
                   _reduction_axes,
                   op_.inInfo(MulOp::getArg1InIndex()),
                   op_.getSettings()) {}

const std::map<int, int> &MulArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), MulOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &MulArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {0, getOutIndex(), GradOpInType::GRADOUT},
      {1, MulOp::getArg0InIndex(), GradOpInType::IN}};
  return inInfo;
}

namespace {
static OpCreator<MulOp> mulOpCreator({Onnx::Operators::Mul_6,
                                      Onnx::Operators::Mul_7});

} // namespace

} // namespace poponnx
