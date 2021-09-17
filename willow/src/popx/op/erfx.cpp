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

void ErfxOpx::grow(poplar::program::Sequence &prog) const {

  const poplar::Tensor &x = getInTensor(ErfOp::getInIndex()).getPoplarTensor();
  auto y                  = popops::erf(graph().getPoplarGraph(), x, prog);
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
