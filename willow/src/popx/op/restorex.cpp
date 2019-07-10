#include <poponnx/error.hpp>
#include <poponnx/op/restore.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/restorex.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <popops/DynamicSlice.hpp>

namespace poponnx {
namespace popx {

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
                           prog);
  prog.add(poplar::program::Copy(actFromStash.squeeze({0}), actToRestore));
  setOutTensor(RestoreOp::getRestoredActOutIndex(), actToRestore);
}

RestoreOpx::RestoreOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RestoreOp>(op);
}

namespace {
OpxCreator<RestoreOpx> restoreOpxCreator(Onnx::CustomOperators::Restore);
} // namespace

} // namespace popx
} // namespace poponnx
