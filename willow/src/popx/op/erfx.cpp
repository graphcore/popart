// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/erf.hpp>
#include <popart/popx/op/erfx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

ErfxOpx::ErfxOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfOp>(op, Onnx::Operators::Erf_9);
}

void ErfxOpx::grow(snap::program::Sequence &prog) const {

  const poplar::Tensor &x = getInTensor(ErfOp::getInIndex()).getPoplarTensor();
  auto y = popops::erf(graph().getPoplarGraph(), x, prog.getPoplarSequence());
  setOutTensor(ErfOp::getOutIndex(), snap::Tensor{y, graph()});
}

ErfxGradOpx::ErfxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfGradOp>(op, Onnx::GradOperators::ErfGrad);
}

// dx erf(x) = 2/Sqrt(Pi) exp(-x^2)
void ErfxGradOpx::grow(snap::program::Sequence &prog) const {
  const poplar::Tensor &x =
      getInTensor(ErfGradOp::getFwdArgInIndex()).getPoplarTensor();
  auto x2 =
      popops::square(graph().getPoplarGraph(), x, prog.getPoplarSequence());
  popops::negInPlace(graph().getPoplarGraph(), x2, prog.getPoplarSequence());
  popops::expInPlace(graph().getPoplarGraph(), x2, prog.getPoplarSequence());
  popops::mulInPlace(
      graph().getPoplarGraph(), x2, 1.1283791671f, prog.getPoplarSequence());

  const poplar::Tensor &gradX =
      getInTensor(ErfGradOp::getGradInIndex()).getPoplarTensor();
  const poplar::Tensor dx = popops::mul(graph().getPoplarGraph(),
                                        x2,
                                        gradX,
                                        prog.getPoplarSequence(),
                                        debugContext("grad_x"));

  setOutTensor(ErfGradOp::getOutIndex(), snap::Tensor{dx, graph()});
}

namespace {
static OpxCreator<ErfxOpx> erfOpxCreator(Onnx::Operators::Erf_9);
static OpxCreator<ErfxGradOpx> erfGradOpxCreator(Onnx::GradOperators::ErfGrad);
} // namespace

} // namespace popx
} // namespace popart
