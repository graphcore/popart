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
  throw error("Grad op has not been implemented for ScaleOp");
}

void ScaleOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

float ScaleOp::getScaleFactor() const { return scale_factor; }

} // namespace poponnx
