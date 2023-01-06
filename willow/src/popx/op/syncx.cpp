// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Program.hpp>
#include <popart/op/sync.hpp>
#include <popart/popx/op/syncx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

SyncOpx::SyncOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SyncOp>(op, Onnx::CustomOperators::Sync);
}

void SyncOpx::grow(poplar::program::Sequence &prog) const {
  auto &syncOp = getOp<SyncOp>();
  prog.add(poplar::program::Sync(syncOp.getSyncType(), debugContext()));
}

namespace {
OpxCreator<SyncOpx> SyncOpxCreator(Onnx::CustomOperators::Sync);
} // namespace

} // namespace popx
} // namespace popart
