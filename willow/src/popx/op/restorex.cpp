#include <popart/error.hpp>
#include <popart/op/restore.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/restorex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
namespace popx {

void RestoreInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto vGraphId     = getOp<RestoreInplaceOp>().getVirtualGraphId();
  auto actToRestore = getInTensor(RestoreInplaceOp::getRestoredActOutIndex());
  auto stash        = getInTensor(RestoreInplaceOp::getStashInIndex());

  auto actFromStash =
      popops::dynamicSlice(graph(),
                           stash,
                           dv_p->pipelineInfo().stashIndex.at(vGraphId),
                           {0},
                           {1},
                           prog,
                           debugPrefix("RestoreInplaceOpx"));
  prog.add(poplar::program::Copy(actFromStash.squeeze({0}), actToRestore));
  setOutTensor(RestoreInplaceOp::getRestoredActOutIndex(), actToRestore);
}

RestoreInplaceOpx::RestoreInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<RestoreInplaceOp>(op);
}

void RestoreOpx::grow(poplar::program::Sequence &prog) const {
  auto vGraphId     = getOp<RestoreOp>().getVirtualGraphId();
  auto actToRestore = getInTensor(RestoreOp::getRestoredActOutIndex());
  auto stash        = getInTensor(RestoreOp::getStashInIndex());

  auto actFromStash =
      popops::dynamicSlice(graph(),
                           stash,
                           dv_p->pipelineInfo().stashIndex.at(vGraphId),
                           {0},
                           {1},
                           prog,
                           debugPrefix("RestoreOpx"));

  setOutTensor(RestoreOp::getRestoredActOutIndex(), actFromStash.squeeze({0}));
}

RestoreOpx::RestoreOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RestoreOp>(op);
}

namespace {
OpxCreator<RestoreOpx> restoreOpxCreator(Onnx::CustomOperators::Restore);
OpxCreator<RestoreInplaceOpx>
    restoreInplaceOpxCreator(Onnx::CustomOperators::RestoreInplace);
} // namespace

} // namespace popx
} // namespace popart
