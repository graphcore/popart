#include <poponnx/makeunique.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

SubtractOp::SubtractOp(const OperatorIdentifier &_opid,
                       Ir *_ir,
                       const std::string &name,
                       const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SubtractOp::clone() const {
  return make_unique<SubtractOp>(*this);
}

std::vector<std::unique_ptr<Op>> SubtractOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = inShape(SubtractOp::getArg0InIndex());
  const auto &shape_o0 = outShape(SubtractOp::getOutIndex());

  upops.emplace_back(make_unique<SubtractArg0GradOp>(
      this, npReductionAxis(shape_a0, shape_o0)));
  upops.emplace_back(make_unique<SubtractArg1GradOp>(this));

  return upops;
}

void SubtractOp::setup() {
  outInfo(SubtractOp::getOutIndex()) =
      npOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

SubtractArg0GradOp::SubtractArg0GradOp(SubtractOp *op_,
                                       const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::SubArg0Grad, op_->pir, _axes, false),
      forward_op_arg_info(op_->inInfo(SubtractOp::getArg0InIndex())) {}

const std::map<int, int> &SubtractArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SubtractOp::getArg0InIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SubtractOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

void SubtractArg0GradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
}

SubtractArg1GradOp::SubtractArg1GradOp(SubtractOp *op_)
    : Op(Onnx::GradOperators::SubArg1Grad, op_->pir),
      forward_op_arg_info(op_->inInfo(SubtractOp::getArg1InIndex())) {}

const std::map<int, int> &SubtractArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SubtractOp::getArg1InIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &SubtractArg1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SubtractOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

std::unique_ptr<Op> SubtractArg1GradOp::clone() const {
  return make_unique<SubtractArg1GradOp>(*this);
}

void SubtractArg1GradOp::setup() {
  outInfo(getOutIndex()) = forward_op_arg_info;
}

namespace {
static OpCreator<SubtractOp> subtractOpCreator({Onnx::Operators::Sub_6,
                                                Onnx::Operators::Sub_7});
static GradOpCreator<SubtractArg0GradOp>
    subtractArg0GradOpCreator(Onnx::GradOperators::SubArg0Grad);
static GradOpCreator<SubtractArg1GradOp>
    subtractArg1GradOpCreator(Onnx::GradOperators::SubArg1Grad);
} // namespace

} // namespace poponnx
