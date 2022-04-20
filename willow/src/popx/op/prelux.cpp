// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/prelu.hpp>
#include <popart/popx/op/prelux.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

namespace pe = popops::expr;

PReluOpx::PReluOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<PReluOp>(op);
}

void PReluOpx::grow(snap::program::Sequence &prog) const {
  auto inputPH = pe::_1;
  auto slopePH = pe::_2;

  // input < 0.0 ? input * slope : input;
  auto expression = pe::Select(
      pe::Mul(inputPH, slopePH), inputPH, pe::Lt(inputPH, pe::Const(0.0f)));

  auto result = snap::popops::map(graph(),
                                  expression,
                                  {getInTensor(PReluOp::getArg0InIndex()),
                                   getInTensor(PReluOp::getArg1InIndex())},
                                  prog,
                                  debugContext("prelu"));

  setOutTensor(PReluOp::getOutIndex(), result);
}

namespace {
OpxCreator<PReluOpx> preluOpxCreator(Onnx::Operators::PRelu_9);
} // namespace

} // namespace popx
} // namespace popart
