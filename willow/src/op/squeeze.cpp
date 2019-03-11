#include <poponnx/makeunique.hpp>
#include <poponnx/op/squeeze.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>

namespace poponnx {

SqueezeOp::SqueezeOp(const OperatorIdentifier &_opid,
                     const std::vector<int64_t> &axes_,
                     const Op::Settings &settings_)
    : Op(_opid, settings_), axes(axes_) {}

void SqueezeOp::setAxesToDefault() {
  auto in_shape = inShape(getInIndex());
  for (int i = 0; i < in_shape.size(); i++) {
    if (in_shape[i] == 1) {
      axes.push_back(i);
    }
  }
}

std::vector<std::unique_ptr<Op>> SqueezeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<SqueezeGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> SqueezeOp::clone() const {
  return make_unique<SqueezeOp>(*this);
}

void SqueezeOp::setup() {
  if (axes.empty()) {
    setAxesToDefault();
  }

  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(),
                            squeeze(inShape(getInIndex()), axes)};
}

void SqueezeOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axes", axes);
}

void SqueezeGradOp::setup() { outInfo(getOutIndex()) = unsqueezedInfo; }

SqueezeGradOp::SqueezeGradOp(const SqueezeOp &op_)
    : Op(Onnx::GradOperators::SqueezeGrad, op_.getSettings()),
      unsqueezedInfo(op_.inInfo(SqueezeOp::getInIndex())) {}

const std::vector<GradInOutMapper> &SqueezeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of squeeze
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), SqueezeOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &SqueezeGradOp::gradOutToNonGradIn() const {
  // the grad-op output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), SqueezeOp::getInIndex()}};
  return outInfo;
}

namespace {
static OpCreator<SqueezeOp> squeezeOpCreator(
    Onnx::Operators::Squeeze_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(new SqueezeOp(_opid, axes, settings));
    },
    true);
} // namespace

} // namespace poponnx
