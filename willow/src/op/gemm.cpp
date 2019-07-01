#include <algorithm>
#include <memory>
#include <poponnx/op/gemm.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/opserialiser.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GemmOp::GemmOp(const OperatorIdentifier &_opid,
               float alpha_,
               float beta_,
               bool transA_,
               bool transB_,
               bool broadcast_,
               const Op::Settings &settings_)
    : Op(_opid, settings_), alpha(alpha_), beta(beta_), transA(transA_),
      transB(transB_), broadcast(broadcast_) {}

std::unique_ptr<Op> GemmOp::clone() const {
  return std::make_unique<GemmOp>(*this);
}

std::vector<std::unique_ptr<Op>> GemmOp::getGradOps() {
  throw error(
      "GemmOp should be removed by pattern 'GemmOp' before call to getGradOps");
}

void GemmOp::setup() {

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

void GemmOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);

  os.appendAttribute("alpha", alpha);
  os.appendAttribute("beta", beta);
  os.appendAttribute("transA", transA);
  os.appendAttribute("transB", transB);

  if (opid.version == 6)
    os.appendAttribute("broadcast", broadcast);
}
namespace {
static OpCreator<GemmOp> gemmOpCreator(
    {Onnx::Operators::Gemm_6, Onnx::Operators::Gemm_7, Onnx::Operators::Gemm_9},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      float alpha = attr.getAttribute<Attributes::Float>("alpha", 1.0);
      float beta  = attr.getAttribute<Attributes::Float>("beta", 1.0);
      bool transA = attr.getAttribute<Attributes::Int>("transA", false);
      bool transB = attr.getAttribute<Attributes::Int>("transB", false);

      // broadcast is only valid for version 6
      bool broadcast = attr.getAttribute<Attributes::Int>("broadcast", false);

      return std::unique_ptr<Op>(
          new GemmOp(_opid, alpha, beta, transA, transB, broadcast, settings));
    },
    true);

} // namespace

} // namespace poponnx
