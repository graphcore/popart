#include <vector>
// for `find', we need the algorithm header
#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

AddOp::AddOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> AddOp::clone() const { return make_unique<AddOp>(*this); }

std::vector<std::unique_ptr<Op>> AddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = inShape(getArg0InIndex());
  const auto &shape_a1 = inShape(getArg1InIndex());
  const auto &shape_o0 = outShape(getOutIndex());

  upops.emplace_back(
      make_unique<AddArg0GradOp>(this, npReductionAxis(shape_a0, shape_o0)));
  upops.emplace_back(
      make_unique<AddArg1GradOp>(this, npReductionAxis(shape_a1, shape_o0)));

  return upops;
}

void AddOp::setup() {
  outInfo(getOutIndex()) =
      npOut(inInfo(getArg0InIndex()), inInfo(getArg1InIndex()));
}

AddArg0GradOp::AddArg0GradOp(AddOp *op_, const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::AddArg0Grad, op_->pir, _axes, false),
      forward_op_arg_info(op_->inInfo(AddOp::getArg0InIndex())) {}

const std::map<int, int> &AddArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg0GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

AddArg1GradOp::AddArg1GradOp(AddOp *op_, const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::GradOperators::AddArg1Grad, op_->pir, _axes, false),
      forward_op_arg_info(op_->inInfo(AddOp::getArg1InIndex())) {}

const std::map<int, int> &AddArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg1GradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of add
  // might need to reduce across certain axes of this
  // if numpy broadcasting happened
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg1GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

namespace {

// Example of how to define the creation method, this will allow you to
// create a more general op that is passed control parameters
static OpCreator<AddOp> addOpCreator(
    Onnx::Operators::Add,
    [](const OperatorIdentifier &opid,
       Ir *ir,
       const std::string &name = "",
       const Attributes &attr  = {}) -> std::unique_ptr<Op> {
      return std::unique_ptr<AddOp>(new AddOp(opid, ir, name, attr));
    },
    true);
static GradOpCreator<AddArg0GradOp>
    addArg0GradOpCreator(Onnx::GradOperators::AddArg0Grad);
static GradOpCreator<AddArg1GradOp>
    addArg1GradOpCreator(Onnx::GradOperators::AddArg1Grad);
} // namespace

} // namespace poponnx
