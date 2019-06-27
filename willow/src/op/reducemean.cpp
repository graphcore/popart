#include <algorithm>
#include <memory>
#include <poponnx/op/reducemean.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceMeanOp::ReduceMeanOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &axes_,
                           const int64_t keepdims_,
                           const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceMeanOp::clone() const {
  return std::make_unique<ReduceMeanOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceMeanOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceMeanGradOp>(*this, backward_shape));
  return result;
}

ReduceMeanGradOp::ReduceMeanGradOp(const ReduceMeanOp &fwdOp,
                                   const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceMeanGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceMeanGradOp::clone() const {
  return std::make_unique<ReduceMeanGradOp>(*this);
}

namespace {
// @SL@ the new factory method for the reduceMean op will get the attributes
// from the model and pass them to the constructor of the OP
static OpCreator<ReduceMeanOp> ReduceMeanOpCreator(
    Onnx::Operators::ReduceMean_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceMeanOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace poponnx
