#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reducesumsquare.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceSumSquareOp::ReduceSumSquareOp(const OperatorIdentifier &_opid,
                                     const std::vector<int64_t> &axes_,
                                     const int64_t keepdims_,
                                     const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceSumSquareOp::clone() const {
  return make_unique<ReduceSumSquareOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceSumSquareOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      make_unique<ReduceSumSquareGradOp>(*this, backward_shape));
  return result;
}

ReduceSumSquareGradOp::ReduceSumSquareGradOp(const ReduceSumSquareOp &fwdOp,
                                             const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceSumSquareGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceSumSquareGradOp::clone() const {
  return make_unique<ReduceSumSquareGradOp>(*this);
}

const std::vector<GradInOutMapper> &
ReduceSumSquareGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceSumSquareOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInInIndex(), ReduceSumSquareOp::getInIndex(), GradOpInType::IN}};

  return inInfo;
}

namespace {
// @SL@ the new factory method for the reduceSumSquare op will get the
// attributes from the model and pass them to the constructor of the OP
static OpCreator<ReduceSumSquareOp> ReduceSumSquareOpCreator(
    Onnx::Operators::ReduceSumSquare_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceSumSquareOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace poponnx
