#include <algorithm>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reduceprod.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ReduceProdOp::ReduceProdOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &axes_,
                           const int64_t keepdims_,
                           const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceProdOp::clone() const {
  return make_unique<ReduceProdOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceProdOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(make_unique<ReduceProdGradOp>(*this, backward_shape));
  return result;
}

ReduceProdGradOp::ReduceProdGradOp(const ReduceProdOp &fwdOp,
                                   const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceProdGrad, fwdOp, backward_shape_),
      axes(fwdOp.getAxes()) {}

std::unique_ptr<Op> ReduceProdGradOp::clone() const {
  return make_unique<ReduceProdGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceProdGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceProdOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdInInIndex(), ReduceProdOp::getInIndex(), GradOpInType::IN}};
  return inInfo;
}

const std::vector<int64_t> &ReduceProdGradOp::getAxes() const { return axes; }

namespace {
// @SL@ the new factory method for the reduceProd op will get the attributes
// from the model and pass them to the constructor of the OP
static OpCreator<ReduceProdOp>
    ReduceProdOpCreator(Onnx::Operators::ReduceProd_1,
                        [](const OperatorIdentifier &_opid,
                           const Op::Settings &settings,
                           const Attributes &attr) -> std::unique_ptr<Op> {
                          int64_t keepdims =
                              attr.getAttribute<Attributes::Int>("keepdims", 1);
                          std::vector<int64_t> axes =
                              attr.getAttribute<Attributes::Ints>("axes", {});

                          return std::unique_ptr<Op>(new ReduceProdOp(
                              _opid, axes, keepdims, settings));
                        },
                        true);
} // namespace

} // namespace poponnx
