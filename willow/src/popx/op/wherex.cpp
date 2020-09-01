// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>

#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/where.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/wherex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

WhereOpx::WhereOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<WhereOp>(op, {Onnx::Operators::Where_9});
}

void WhereOpx::grow(poplar::program::Sequence &prog) const {

  auto condition = getInTensor(WhereOp::conditionInIndex());
  auto x         = getInTensor(WhereOp::xInIndex());
  auto y         = getInTensor(WhereOp::yInIndex());

  auto result = popops::select(
      graph(), x, y, condition, prog, debugPrefix(), poplar::OptionFlags());

  setOutTensor(WhereOp::outIndex(), result);
}

namespace {
OpxCreator<WhereOpx> whereOpxCreator(Onnx::Operators::Where_9);
} // namespace

} // namespace popx
} // namespace popart
