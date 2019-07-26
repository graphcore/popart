#include <popart/error.hpp>
#include <popart/op/stash.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/stashx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <popops/DynamicSlice.hpp>

namespace popart {
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
                        prog,
                        debugPrefix("stash"));

  setOutTensor(StashOp::getOutIndex(), outTensor);
}

StashOpx::StashOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<StashOp>(op);
}

namespace {
OpxCreator<StashOpx> stashOpxCreator(Onnx::CustomOperators::Stash);
} // namespace

} // namespace popx
} // namespace popart
