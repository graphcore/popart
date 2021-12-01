// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

void StashOpx::growStaticStashUpdate(snap::program::Sequence &prog,
                                     const snap::Tensor &stashIndex,
                                     const snap::Tensor &inTensor,
                                     const snap::Tensor &outTensor) const {
  /*
    We cannot do a dynamic update based on tensor stashIndex, but we can do a
    dynamic switch-case on stashIndex. There are hStashSize cases, with each
    case i being the program that should be run if stashIndex has the value i.
    We have thus "unrolled" the dynamic update in a way.
  */

  poplar::program::Switch switchCase(stashIndex.reshape({}).getPoplarTensor(),
                                     debugContext("static-stash/switch"));

  for (unsigned i = 0; i != hStashSize; ++i) {
    const auto outSliceAtIdx = outTensor.slice(i, i + 1, 0);
    switchCase.add(i,
                   poplar::program::Copy(inTensor.getPoplarTensor(),
                                         outSliceAtIdx.getPoplarTensor(),
                                         false,
                                         debugContext("static-stash/switch-" +
                                                      std::to_string(i))));
  }

  prog.getPoplarSequence().add(switchCase);
}

void StashOpx::growDynamicStashUpdate(snap::program::Sequence &prog,
                                      const snap::Tensor &stashIndex,
                                      const snap::Tensor &inTensor,
                                      const snap::Tensor &outTensor) const {
  // Update the stash.
  popops::dynamicUpdate(graph().getPoplarGraph(),
                        outTensor.getPoplarTensor(),
                        inTensor.expand({0}).getPoplarTensor(),
                        stashIndex.getPoplarTensor(),
                        {0},
                        {1},
                        prog.getPoplarSequence(),
                        debugContext("stash"));
}

void StashOpx::grow(snap::program::Sequence &prog) const {
  // Create the stash size tensor.
  const auto stashSize =
      getConst(poplar::UNSIGNED_INT, {}, hStashSize, "stash_size");

  // Create the stash index tensor.
  const snap::Tensor stashIndex =
      snap::Tensor{graph().getPoplarGraph().addVariable(
                       poplar::UNSIGNED_INT, {1}, debugContext("stash_index")),
                   graph()};
  graph().getPoplarGraph().setTileMapping(stashIndex.getPoplarTensor(), 0);
  graph().getPoplarGraph().setInitialValue(stashIndex.getPoplarTensor(),
                                           poplar::ArrayRef<uint32_t>({0}));

  // Retrieve the input tensor.
  const auto &inTensor = getInTensor(StashOp::getInIndex());

  // Create the output tensor.
  const auto outTensor =
      snap::Tensor{popops::createSliceableTensorFromSlice(
                       graph().getPoplarGraph(),
                       inTensor.expand({0}).getPoplarTensor(),
                       {0},
                       {hStashSize},
                       outId(StashOp::getOutIndex())),
                   graph()};

  // Create the stash tensor (the output) and grow the program to update it.
  if (canDynamicUpdateStash) {
    growDynamicStashUpdate(prog, stashIndex, inTensor, outTensor);
  } else {
    growStaticStashUpdate(prog, stashIndex, inTensor, outTensor);
  }
  setOutTensor(StashOp::getOutIndex(), outTensor);

  // Create a "1" tensor and grow program to increment stash index by 1.
  const auto one =
      getConst(poplar::UNSIGNED_INT, {}, 1.0, "one").getPoplarTensor();
  popops::addInPlace(graph().getPoplarGraph(),
                     stashIndex.getPoplarTensor(),
                     one,
                     prog.getPoplarSequence());
  popops::remInPlace(graph().getPoplarGraph(),
                     stashIndex.getPoplarTensor(),
                     stashSize.getPoplarTensor(),
                     prog.getPoplarSequence());
}

StashOpx::StashOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<StashOp>(op);
  hStashSize = static_cast<size_t>(getOp<StashOp>().getStashSize());
  // INT8/UINT8 now supported. Leaving the fallbacks until stashx/restorex
  // will get removed wholesale
  // TODO: T51331
  canDynamicUpdateStash = true;
}

namespace {
OpxCreator<StashOpx> stashOpxCreator(Onnx::CustomOperators::Stash);
} // namespace

} // namespace popx
} // namespace popart
