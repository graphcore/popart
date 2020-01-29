#include <popart/error.hpp>
#include <popart/op/stash.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/stashx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

void StashOpx::grow(poplar::program::Sequence &prog) const {
  auto &stashOp = getOp<StashOp>();
  auto outInfo  = stashOp.outInfo(StashOp::getOutIndex());

  std::vector<poplar::Tensor> stashes;
  for (int64_t i = 0; i < stashOp.getStashSize(); i++) {
    auto stashPart =
        graph().clone(getInTensor(StashOp::getInIndex()).expand({0}),
                      "Stash__" + inId(StashOp::getInIndex()));
    stashes.push_back(stashPart);
  }
  auto outTensor = poplar::concat(stashes, 0);

  // Create the stash index tensor
  auto one       = getConst(poplar::UNSIGNED_INT, {}, 1.0, debugPrefix("one"));
  auto stashSize = getConst(poplar::UNSIGNED_INT,
                            {},
                            stashOp.getStashSize(),
                            debugPrefix("stash_size"));

  poplar::Tensor stashIndex = graph().addVariable(poplar::UNSIGNED_INT, {1});
  graph().setTileMapping(stashIndex, 0);
  graph().setInitialValue(stashIndex, poplar::ArrayRef<uint32_t>({0}));

  // Update the stash
  popops::dynamicUpdate(graph(),
                        outTensor,
                        getInTensor(StashOp::getInIndex()).expand({0}),
                        stashIndex,
                        {0},
                        {1},
                        prog,
                        debugPrefix("stash"));

  // Increment the stash index
  popops::addInPlace(graph(), stashIndex, one, prog);
  popops::remInPlace(graph(), stashIndex, stashSize, prog);

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
