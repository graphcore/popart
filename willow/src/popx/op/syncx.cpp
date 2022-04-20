// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <snap/Program.hpp>
#include <popart/op/sync.hpp>
#include <popart/popx/op/syncx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

SyncOpx::SyncOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<SyncOp>(op, Onnx::CustomOperators::Sync);
}

void SyncOpx::grow(snap::program::Sequence &prog) const {
  auto &syncOp = getOp<SyncOp>();
  prog.add(snap::program::Sync(graph(), syncOp.getSyncType(), debugContext()));
}

namespace {
OpxCreator<SyncOpx> SyncOpxCreator(Onnx::CustomOperators::Sync);
} // namespace

} // namespace popx
} // namespace popart
