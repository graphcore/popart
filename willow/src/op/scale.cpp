#include <memory>
#include <poponnx/op/scale.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

std::vector<std::tuple<OperatorIdentifier, float>>
ScaleOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::ScaleInplace, 10}};
}

ScaleInplaceOp::ScaleInplaceOp(const ScaleOp &scale_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::ScaleInplace,
                                scale_op.getSettings()),
      scale_factor(scale_op.getScaleFactor()) {}

std::unique_ptr<Op>
ScaleOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ScaleInplace) {
    return std::make_unique<ScaleInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

ScaleOp::ScaleOp(const OperatorIdentifier &_opid,
                 float scale_,
                 const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), scale_factor(scale_) {}

std::unique_ptr<Op> ScaleOp::clone() const {
  return std::make_unique<ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScaleOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<ScaleGradOp>(*this));
  return upops;
}

float ScaleOp::getScaleFactor() const { return scale_factor; }
float ScaleInplaceOp::getScaleFactor() const { return scale_factor; }

void ScaleOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("scale", scale_factor);
}

void ScaleInplaceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("scale", scale_factor);
}

std::unique_ptr<Op> ScaleInplaceOp::clone() const {
  return std::make_unique<ScaleInplaceOp>(*this);
}

// A scale with a scale factor of +1 can be replaced by identity
bool ScaleOp::canBeReplacedByIdentity() { return getScaleFactor() == 1.0f; }

ScaleGradOp::ScaleGradOp(const ScaleOp &fwdOp)
    : ScaleOp(Onnx::GradOperators::ScaleGrad,
              fwdOp.getScaleFactor(),
              fwdOp.getSettings()) {}

std::unique_ptr<Op> ScaleGradOp::clone() const {
  return std::make_unique<ScaleGradOp>(*this);
}

const std::vector<GradInOutMapper> &ScaleGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ScaleOp::getOutIndex(), GradOpInType::GRADOUT}};

  return inInfo;
}

const std::map<int, int> &ScaleGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ScaleOp::getInIndex()}};

  return outInfo;
}

namespace {
static OpCreator<ScaleOp> scaleOpCreator(
    Onnx::CustomOperators::Scale_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float scale = attr.getAttribute<Attributes::Float>("scale", 1.0f);

      return std::unique_ptr<Op>(new ScaleOp(_opid, scale, settings));
    },
    true);

} // namespace
} // namespace poponnx
