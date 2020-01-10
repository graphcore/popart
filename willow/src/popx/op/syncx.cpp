#include <popart/error.hpp>
#include <popart/op/sync.hpp>
#include <popart/popx/op/syncx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SyncOpx::SyncOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SyncOp>(op, Onnx::CustomOperators::Sync);
}

void SyncOpx::grow(poplar::program::Sequence &prog) const {
  auto &syncOp = getOp<SyncOp>();
  prog.add(poplar::program::Sync(syncOp.getSyncType()));
}

namespace {
OpxCreator<SyncOpx> SyncOpxCreator(Onnx::CustomOperators::Sync);
} // namespace

} // namespace popx
} // namespace popart
