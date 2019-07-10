#include <poponnx/error.hpp>
#include <poponnx/op/stash.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/stashx.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensor.hpp>

#include <popops/DynamicSlice.hpp>

namespace poponnx {
namespace popx {

void StashOpx::grow(poplar::program::Sequence &prog) const {
  auto vGraphId  = getOp<StashOp>().getVirtualGraphId();
  auto outInfo   = getOp<StashOp>().outInfo(StashOp::getOutIndex());
  auto outTensor = popops::createSliceableTensor(
      graph(), popType(outInfo), outInfo.shape_szt(), {0}, {1});

  popops::dynamicUpdate(graph(),
                        outTensor,
                        getInTensor(StashOp::getInIndex()).expand({0}),
                        dv_p->pipelineInfo().stashIndex.at(vGraphId),
                        {0},
                        {1},
                        prog);

  setOutTensor(StashOp::getOutIndex(), outTensor);
}

StashOpx::StashOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<StashOp>(op);
}

namespace {
OpxCreator<StashOpx> stashOpxCreator(Onnx::CustomOperators::Stash);
} // namespace

} // namespace popx
} // namespace poponnx
