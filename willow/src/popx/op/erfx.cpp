// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/erf.hpp>
#include <popart/popx/op/erfx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

ErfxOpx::ErfxOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfOp>(op, Onnx::Operators::Erf_9);
}

void ErfxOpx::grow(poplar::program::Sequence &prog) const {

  const poplar::Tensor &x = getInTensor(ErfOp::getInIndex());
  auto y                  = popops::erf(graph(), x, prog);
  setOutTensor(ErfOp::getOutIndex(), y);
}

ErfxGradOpx::ErfxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ErfGradOp>(op, Onnx::GradOperators::ErfGrad);
}

// dx erf(x) = 2/Sqrt(Pi) exp(-x^2)
void ErfxGradOpx::grow(poplar::program::Sequence &prog) const {
  const poplar::Tensor &x = getInTensor(ErfGradOp::getFwdArgInIndex());
  auto x2                 = popops::square(graph(), x, prog);
  popops::negInPlace(graph(), x2, prog);
  popops::expInPlace(graph(), x2, prog);
  popops::mulInPlace(graph(), x2, 1.1283791671f, prog);

  const poplar::Tensor &gradX = getInTensor(ErfGradOp::getGradInIndex());
  const poplar::Tensor dx =
      popops::mul(graph(), x2, gradX, prog, debugContext("grad_x"));

  setOutTensor(ErfGradOp::getOutIndex(), dx);
}

namespace {
static OpxCreator<ErfxOpx> erfOpxCreator(Onnx::Operators::Erf_9);
static OpxCreator<ErfxGradOpx> erfGradOpxCreator(Onnx::GradOperators::ErfGrad);
} // namespace

} // namespace popx
} // namespace popart
