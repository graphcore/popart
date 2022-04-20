// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <poplar/ArrayRef.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/restore.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/restorex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {

template <typename Derived>
RestoreBaseOpx<Derived>::RestoreBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  // Note RestoreInplaceOp derives RestoreOp.
  verifyOp<typename Derived::OpType>(op);

  // INT8/UINT8 now supported. Leaving the fallbacks until stashx/restorex
  // will get removed wholesale
  // TODO: T51331
  canDynamicSliceRestore = true;
}

template <typename Derived>
snap::Tensor
RestoreBaseOpx<Derived>::growRestore(snap::program::Sequence &prog,
                                     const snap::Tensor &stash) const {
  const auto &op       = getOp<typename Derived::OpType>();
  const auto stashSize = op.getStashSize();

  // Create the stash index tensor.
  const auto stashIndex =
      graph().addVariable(poplar::UNSIGNED_INT, {1}, debugContext());
  graph().getPoplarGraph().setTileMapping(stashIndex.getPoplarTensor(), 0);
  graph().getPoplarGraph().setInitialValue(stashIndex.getPoplarTensor(),
                                           poplar::ArrayRef<uint32_t>({0}));
  dv_p->lowering().addPipelineIndexTensor(stashIndex);

  // Create the stash size tensor.
  const auto stashSizeTensor =
      getConst(poplar::UNSIGNED_INT, {}, stashSize, "stash_size")
          .getPoplarTensor();

  // Grow program to take slice of stash at the stash index.
  snap::Tensor actFromStash;

  if (canDynamicSliceRestore) {
    actFromStash = growDynamicSliceRestore(prog, stashIndex, stash);
  } else {
    actFromStash = growStaticSliceRestore(prog, stashSize, stashIndex, stash);
  }

  // Create a "1" tensor and grow program to increment stash index by 1.
  auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one");
  snap::popops::addInPlace(graph(), stashIndex, one, prog, debugContext());
  popops::remInPlace(graph().getPoplarGraph(),
                     stashIndex.getPoplarTensor(),
                     stashSizeTensor,
                     prog.getPoplarSequence(),
                     debugContext());

  return actFromStash;
}

template <typename Derived>
snap::Tensor RestoreBaseOpx<Derived>::growStaticSliceRestore(
    snap::program::Sequence &prog,
    const int64_t stashSize,
    const snap::Tensor &stashIndex,
    const snap::Tensor &stash) const {

  // stash is (N, a, b, c). Output is (a, b, c) at index stashIndex.

  // Creates (1, a, b, c) tensor.
  snap::Tensor actFromStash = {
      popops::createSliceTensor(graph().getPoplarGraph(),
                                stash.getPoplarTensor(),
                                {0},
                                {1},
                                1,
                                debugContext("static-restore/out-slice")),
      graph()};

  snap::program::Switch switchCase(
      stashIndex.reshape({}),
      debugContext("static-restore/switch-on-stash-index"));

  for (int64_t i = 0; i < stashSize; i++) {
    const auto inSliceAtIdx = stash.slice(i, i + 1, 0);
    switchCase.add(
        i,
        snap::program::Copy(
            inSliceAtIdx,
            actFromStash,
            false,
            debugContext("static-restore/switch-copy-" + std::to_string(i))));
  }

  prog.add(switchCase);

  return actFromStash.squeeze({0});
}

template <typename Derived>
snap::Tensor RestoreBaseOpx<Derived>::growDynamicSliceRestore(
    snap::program::Sequence &prog,
    const snap::Tensor &stashIndex,
    const snap::Tensor &stash) const {

  auto actFromStash =
      popops::dynamicSlice(graph().getPoplarGraph(),
                           stash.getPoplarTensor(),
                           stashIndex.getPoplarTensor(),
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           debugContext("grow_restore_dynamic_slice"));

  return snap::Tensor{actFromStash.squeeze({0}), graph()};
}

void RestoreInplaceOpx::grow(snap::program::Sequence &prog) const {
  auto actToRestore = getInTensor(RestoreInplaceOp::getActToRestoreInIndex());
  auto stash        = getInTensor(RestoreInplaceOp::getStashInIndex());

  const auto actFromStash = growRestore(prog, stash);

  prog.add(
      snap::program::Copy(actFromStash, actToRestore, false, debugContext()));
  setOutTensor(RestoreInplaceOp::getRestoredActOutIndex(), actToRestore);
}

RestoreInplaceOpx::RestoreInplaceOpx(Op *op, Devicex *devicex)
    : RestoreBaseOpx(op, devicex) {}

void RestoreOpx::grow(snap::program::Sequence &prog) const {
  auto stash = getInTensor(RestoreOp::getStashInIndex());

  auto actFromStash = growRestore(prog, stash);

  setOutTensor(RestoreOp::getRestoredActOutIndex(), actFromStash);
}

RestoreOpx::RestoreOpx(Op *op, Devicex *devicex)
    : RestoreBaseOpx(op, devicex) {}

namespace {
OpxCreator<RestoreOpx> restoreOpxCreator(Onnx::CustomOperators::Restore);
OpxCreator<RestoreInplaceOpx>
    restoreInplaceOpxCreator(Onnx::CustomOperators::RestoreInplace);
} // namespace

} // namespace popx
} // namespace popart
