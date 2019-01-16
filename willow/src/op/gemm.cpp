#include <algorithm>
#include <poponnx/logging.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/gemm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GemmOp::GemmOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name,
               const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> GemmOp::clone() const { return make_unique<GemmOp>(*this); }

std::vector<std::unique_ptr<Op>> GemmOp::getGradOps() {
  throw error(
      "GemmOp should be removed by pattern 'GemmOp' before call to getGradOps");
}

void GemmOp::setup() {
  // override defaults if present
  nAtts.setIfPresent(alpha, "alpha");
  nAtts.setIfPresent(beta, "beta");
  nAtts.setIfPresent(transA, "transA");
  nAtts.setIfPresent(transB, "transB");

  outInfo(getOutIndex()) = {inInfo(getAInIndex()).dataType(), getOutputShape()};
}

Shape GemmOp::getOutputShape() {
  auto a_shape = inInfo(getAInIndex()).shape();
  if (transA) {
    std::reverse(a_shape.begin(), a_shape.end());
  }

  auto b_shape = inInfo(getBInIndex()).shape();
  if (transB) {
    std::reverse(b_shape.begin(), b_shape.end());
  }

  return {a_shape[0], b_shape[1]};
}

float GemmOp::getAlpha() const { return alpha; }

float GemmOp::getBeta() const { return beta; }

bool GemmOp::getTransA() const { return transA; }
bool GemmOp::getTransB() const { return transB; }

namespace {
static OpCreator<GemmOp> gemmOpCreator({Onnx::Operators::Gemm_6,
                                        Onnx::Operators::Gemm_7,
                                        Onnx::Operators::Gemm_9});
} // namespace

} // namespace poponnx
