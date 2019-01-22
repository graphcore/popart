#include <poponnx/makeunique.hpp>
#include <poponnx/op/unsqueeze.hpp>
#include <poponnx/opmanager.hpp>

namespace poponnx {

UnsqueezeOp::UnsqueezeOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &axes_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_) {}

std::vector<std::unique_ptr<Op>> UnsqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<UnsqueezeGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> UnsqueezeOp::clone() const {
  return make_unique<UnsqueezeOp>(*this);
}

void UnsqueezeOp::setup() {
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            unsqueeze(inShape(getInIndex()), axes)};
}

void UnsqueezeOp::appendAttributes(std::stringstream &ss,
                                   const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "axes", axes);
}

void UnsqueezeGradOp::setup() { outInfo(getOutIndex()) = squeezedInfo; }

UnsqueezeGradOp::UnsqueezeGradOp(const UnsqueezeOp &op_)
    : Op(Onnx::GradOperators::UnsqueezeGrad, op_.getSettings()),
      squeezedInfo(op_.inInfo(UnsqueezeOp::getInIndex())) {}

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
static OpCreator<UnsqueezeOp> unsqueezeOpCreator(
    Onnx::Operators::Unsqueeze_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(new UnsqueezeOp(_opid, axes, settings));
    },
    true);
} // namespace

} // namespace poponnx
