#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reducelogsum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceLogSumOp::ReduceLogSumOp(const OperatorIdentifier &_opid,
                               const std::vector<int64_t> &axes_,
                               const int64_t keepdims_,
                               const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceLogSumOp::clone() const {
  return make_unique<ReduceLogSumOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceLogSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(make_unique<ReduceLogSumGradOp>(*this, backward_shape));
  return result;
}

ReduceLogSumGradOp::ReduceLogSumGradOp(const ReduceLogSumOp &fwdOp,
                                       const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceLogSumGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceLogSumGradOp::clone() const {
  return make_unique<ReduceLogSumGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceLogSumGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceLogSumGradOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(),
       ReduceLogSumGradOp::getOutIndex(),
       GradOpInType::OUT}};

  return inInfo;
}

namespace {
// @SL@ the new factory method for the reduceLogSum op will get the attributes
// from the model and pass them to the constructor of the OP
static OpCreator<ReduceLogSumOp> ReduceLogSumOpCreator(
    Onnx::Operators::ReduceLogSum_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceLogSumOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace poponnx
