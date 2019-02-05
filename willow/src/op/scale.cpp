#include <poponnx/makeunique.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ScaleOp::ScaleOp(const OperatorIdentifier &_opid,
                 float scale_,
                 const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), scale_factor(scale_) {}

std::unique_ptr<Op> ScaleOp::clone() const {
  return make_unique<ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScaleOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ScaleGradOp>(*this));
  return upops;
}

float ScaleOp::getScaleFactor() const { return scale_factor; }

void ScaleOp::appendAttributes(std::stringstream &ss,
                               const std::string &tab) const {
  Op::appendAttributes(ss, tab);
  appendAttribute(ss, tab, "scale", scale_factor);
}
ScaleGradOp::ScaleGradOp(const ScaleOp &fwdOp)
    : ScaleOp(Onnx::GradOperators::ScaleGrad,
              fwdOp.getScaleFactor(),
              fwdOp.getSettings()) {}

std::unique_ptr<Op> ScaleGradOp::clone() const {
  return make_unique<ScaleGradOp>(*this);
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
    Onnx::Operators::Scale_1,
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float scale = attr.getAttribute<Attributes::Float>("scale", 1.0f);

      return std::unique_ptr<Op>(new ScaleOp(_opid, scale, settings));
    },
    true);
// static GradOpCreator<ScaleGradOp>
//    scaleGradOpCreator(Onnx::GradOperators::ScaleGrad);

} // namespace
} // namespace poponnx
