#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reducemax.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceMaxOp::ReduceMaxOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &axes_,
                         const int64_t keepdims_,
                         const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceMaxOp::clone() const {
  return make_unique<ReduceMaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceMaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(make_unique<ReduceMaxGradOp>(*this, backward_shape));
  return result;
}

ReduceMaxGradOp::ReduceMaxGradOp(const ReduceMaxOp &fwdOp,
                                 const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceMaxGrad, fwdOp, backward_shape_) {
}

std::unique_ptr<Op> ReduceMaxGradOp::clone() const {
  return make_unique<ReduceMaxGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceMaxGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceMaxOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInInIndex(), ReduceMaxOp::getInIndex(), GradOpInType::IN},
      {getFwdOutInIndex(), ReduceMaxOp::getOutIndex(), GradOpInType::OUT}};
  return inInfo;
}

namespace {
// @SL@ the new factory method for the reduceMax op will get the attributes from
// the model and pass them to the constructor of the OP
static OpCreator<ReduceMaxOp>
    reduceMaxOpCreator(Onnx::Operators::ReduceMax_1,
                       [](const OperatorIdentifier &_opid,
                          const Op::Settings &settings,
                          const Attributes &attr) -> std::unique_ptr<Op> {
                         int64_t keepdims =
                             attr.getAttribute<Attributes::Int>("keepdims", 1);
                         std::vector<int64_t> axes =
                             attr.getAttribute<Attributes::Ints>("axes", {});

                         return std::unique_ptr<Op>(
                             new ReduceMaxOp(_opid, axes, keepdims, settings));
                       },
                       true);
} // namespace

} // namespace poponnx
