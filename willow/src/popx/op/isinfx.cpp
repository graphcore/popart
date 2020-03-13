// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/isinf.hpp>
#include <popart/popx/op/isinfx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

IsInfx::IsInfx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IsInf>(op, Onnx::Operators::IsInf_10);
}

void IsInfx::grow(poplar::program::Sequence &prog) const {
  // (x == x) && x !isFinite
  setOutTensor(
      IsInf::getOutIndex(),
      popops::map(
          graph(),
          popops::expr::And(
              popops::expr::Equal(popops::expr::_1, popops::expr::_1),
              popops::expr::Not(popops::expr::IsFinite(popops::expr::_1))),
          {get(inId(0))},
          prog,
          debugPrefix()));
}

namespace {
OpxCreator<IsInfx> IsInfxCreator(Onnx::Operators::IsInf_10);
} // namespace

} // namespace popx
} // namespace popart
