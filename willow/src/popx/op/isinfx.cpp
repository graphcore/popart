// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popart/op/isinf.hpp>
#include <popart/popx/op/isinfx.hpp>
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

IsInfx::IsInfx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IsInf>(op, Onnx::Operators::IsInf_10);
}

void IsInfx::grow(snap::program::Sequence &prog) const {
  // (x == x) && x !isFinite
  setOutTensor(
      IsInf::getOutIndex(),
      snap::popops::map(
          graph(),
          popops::expr::And(
              popops::expr::Equal(popops::expr::_1, popops::expr::_1),
              popops::expr::Not(popops::expr::IsFinite(popops::expr::_1))),
          {get(inId(0))},
          prog,
          debugContext()));
}

namespace {
OpxCreator<IsInfx> IsInfxCreator(Onnx::Operators::IsInf_10);
} // namespace

} // namespace popx
} // namespace popart
