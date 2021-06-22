// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/erf.hpp>
#include <popart/popx/op/erfx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ErfxOpx::ErfxOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfOp>(op, Onnx::Operators::Erf_9);
}

// erf(x) ≈ 1 - (c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5)exp(-x^2)
// where t = 1/(1 + con*x) , for x>= 0. And erf(x) = −erf(−x).
// Approximation formula, maximum error: 1.5×10^−7
// https://en.wikipedia.org/wiki/Error_function, Numerical approximations.
void ErfxOpx::grow(poplar::program::Sequence &prog) const {

  const poplar::Tensor &x = getInTensor(ErfOp::getInIndex()).getPoplarTensor();

  poplar::Tensor sign = popops::signum(graph().getPoplarGraph(), x, prog);
  poplar::Tensor y    = popops::abs(graph().getPoplarGraph(), x, prog);
  popops::mulInPlace(graph().getPoplarGraph(), y, 0.3275911f, prog);
  popops::addInPlace(graph().getPoplarGraph(), y, 1.0f, prog);
  popops::invInPlace(graph().getPoplarGraph(), y, prog);

  static const std::array<float, 4> coeff{
      -1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f};
  poplar::Tensor poly =
      popops::mul(graph().getPoplarGraph(), y, 1.061405429f, prog);
  for (float c : coeff) {
    popops::addInPlace(graph().getPoplarGraph(), poly, c, prog);
    popops::mulInPlace(graph().getPoplarGraph(), poly, y, prog);
  }

  y = popops::square(graph().getPoplarGraph(), x, prog);
  popops::negInPlace(graph().getPoplarGraph(), y, prog);
  popops::expInPlace(graph().getPoplarGraph(), y, prog);
  popops::mulInPlace(graph().getPoplarGraph(), y, poly, prog);
  popops::negInPlace(graph().getPoplarGraph(), y, prog);
  popops::addInPlace(graph().getPoplarGraph(), y, 1.0f, prog);
  popops::mulInPlace(graph().getPoplarGraph(), y, sign, prog);

  setOutTensor(ErfOp::getOutIndex(), snap::Tensor{y, graph()});
}

ErfxGradOpx::ErfxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfGradOp>(op, Onnx::GradOperators::ErfGrad);
}

// dx erf(x) = 2/Sqrt(Pi) exp(-x^2)
void ErfxGradOpx::grow(poplar::program::Sequence &prog) const {
  const poplar::Tensor &x =
      getInTensor(ErfGradOp::getFwdArgInIndex()).getPoplarTensor();
  auto x2 = popops::square(graph().getPoplarGraph(), x, prog);
  popops::negInPlace(graph().getPoplarGraph(), x2, prog);
  popops::expInPlace(graph().getPoplarGraph(), x2, prog);
  popops::mulInPlace(graph().getPoplarGraph(), x2, 1.1283791671f, prog);

  const poplar::Tensor &gradX =
      getInTensor(ErfGradOp::getGradInIndex()).getPoplarTensor();
  const poplar::Tensor dx = popops::mul(
      graph().getPoplarGraph(), x2, gradX, prog, debugContext("grad_x"));

  setOutTensor(ErfGradOp::getOutIndex(), snap::Tensor{dx, graph()});
}

namespace {
static OpxCreator<ErfxOpx> erfOpxCreator(Onnx::Operators::Erf_9);
static OpxCreator<ErfxGradOpx> erfGradOpxCreator(Onnx::GradOperators::ErfGrad);
} // namespace

} // namespace popx
} // namespace popart
