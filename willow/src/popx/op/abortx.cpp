// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Program.hpp>
#include <popart/op/abort.hpp>
#include <popart/popx/op/abortx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

AbortOpx::AbortOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AbortOp>(op, Onnx::CustomOperators::Abort);
}

void AbortOpx::grow(poplar::program::Sequence &prog) const {
  if (hasInput(AbortOp::getInIndex())) {
    auto condition = getInTensor(AbortOp::getInIndex());
    prog.add(poplar::program::AbortOnCondition(condition));
  } else {
    prog.add(poplar::program::Abort());
  }
}

namespace {
OpxCreator<AbortOpx> AbortxCreator(Onnx::CustomOperators::Abort);
} // namespace

} // namespace popx
} // namespace popart
