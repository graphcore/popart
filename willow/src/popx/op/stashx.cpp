// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <poplar/ArrayRef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/stash.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/stashx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

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

  snap::program::Switch switchCase(stashIndex.reshape({}),
                                   debugContext("static-stash/switch"));

  for (unsigned i = 0; i != hStashSize; ++i) {
    const auto outSliceAtIdx = outTensor.slice(i, i + 1, 0);
    switchCase.add(i,
                   snap::program::Copy(inTensor,
                                       outSliceAtIdx,
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
      getScalarVariable(poplar::UNSIGNED_INT, "stash_index").reshape({1});
  graph().getPoplarGraph().setInitialValue(stashIndex.getPoplarTensor(),
                                           poplar::ArrayRef<uint32_t>({0}));
  dv_p->lowering().addPipelineIndexTensor(stashIndex);

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
  const auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one");
  snap::popops::addInPlace(graph(), stashIndex, one, prog, debugContext());
  popops::remInPlace(graph().getPoplarGraph(),
                     stashIndex.getPoplarTensor(),
                     stashSize.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     debugContext());
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
