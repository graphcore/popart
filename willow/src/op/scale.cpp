#include <poponnx/makeunique.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ScaleOp::ScaleOp(const OperatorIdentifier &_opid,
                 Ir *_ir,
                 const std::string &name,
                 const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> ScaleOp::clone() const {
  return make_unique<ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScaleOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ScaleGradOp>(this));
  return upops;
}

float ScaleOp::getScaleFactor() const { return scale_factor; }

ScaleGradOp::ScaleGradOp(ScaleOp *fwdOp)
    : ScaleOp(Onnx::GradOperators::ScaleGrad, fwdOp->pir) {
  setScaleFactor(fwdOp->getScaleFactor());
}

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
static OpCreator<ScaleOp> scaleOpCreator(Onnx::Operators::Scale);
static GradOpCreator<ScaleGradOp>
    scaleGradOpCreator(Onnx::GradOperators::ScaleGrad);

} // namespace
} // namespace poponnx
