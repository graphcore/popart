#include <poponnx/makeunique.hpp>
#include <poponnx/op/unsqueeze.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

UnsqueezeOp::UnsqueezeOp(const OperatorIdentifier &_opid,
                         Ir *_ir,
                         const std::string &name,
                         const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  _attr.setIfPresent(axes, "axes");
}

std::vector<std::unique_ptr<Op>> UnsqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<UnsqueezeGradOp>(this));
  return upops;
}

std::unique_ptr<Op> UnsqueezeOp::clone() const {
  return make_unique<UnsqueezeOp>(*this);
}

void UnsqueezeOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            unsqueeze(inShape(getInIndex()), axes)};
}

void UnsqueezeGradOp::setup() { outInfo(getOutIndex()) = squeezedInfo; }

UnsqueezeGradOp::UnsqueezeGradOp(UnsqueezeOp *op_)
    : Op(Onnx::GradOperators::UnsqueezeGrad, op_->pir),
      squeezedInfo(op_->inInfo(UnsqueezeOp::getInIndex())) {}

const std::vector<GradInOutMapper> &UnsqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of unsqueeze
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), UnsqueezeOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &UnsqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), UnsqueezeOp::getInIndex()}};
  return outInfo;
}

namespace {
static OpCreator<UnsqueezeOp> unsqueezeOpCreator(Onnx::Operators::Unsqueeze_1);
static GradOpCreator<UnsqueezeGradOp>
    unsqueezeGradOpCreator(Onnx::GradOperators::UnsqueezeGrad);
} // namespace

} // namespace poponnx
