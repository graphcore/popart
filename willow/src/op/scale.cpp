#include <poponnx/makeunique.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ScaleOp::ScaleOp(const OpConstructorBundle &bundle, float scale_factor_)
    : Op(bundle), scale_factor(scale_factor_) {}

std::unique_ptr<Op> ScaleOp::clone() const {
  return make_unique<ScaleOp>(*this);
}

std::vector<std::unique_ptr<Op>> ScaleOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ScaleGradOp>(this));
  return upops;
}

void ScaleOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

float ScaleOp::getScaleFactor() const { return scale_factor; }

ScaleGradOp::ScaleGradOp(ScaleOp *fwdOp)
    : ScaleOp({"ScaleGrad", fwdOp->pir, {}, getPoponnxDomain()},
              fwdOp->getScaleFactor()) {}

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

} // namespace poponnx
