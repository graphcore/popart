// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <string>
#include <poplar/ArrayRef.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popart/op/restore.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/restorex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"

namespace popart {
class Op;

namespace popx {

template <typename Derived>
RestoreBaseOpx<Derived>::RestoreBaseOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  // Note RestoreInplaceOp derives RestoreOp.
  verifyOp<typename Derived::OpType>(op);

  // INT8/UINT8 now supported. Leaving the fallbacks until stashx/restorex
  // will get removed wholesale
  // TODO: T51331
  canDynamicSliceRestore = true;
}

template <typename Derived>
poplar::Tensor
RestoreBaseOpx<Derived>::growRestore(poplar::program::Sequence &prog,
                                     const poplar::Tensor &stash) const {
  const auto &op       = getOp<typename Derived::OpType>();
  const auto stashSize = op.getStashSize();

  // Create the stash index tensor.
  const auto stashIndex =
      getScalarVariable(poplar::UNSIGNED_INT, "stash_index").reshape({1});
  graph().setInitialValue(stashIndex, poplar::ArrayRef<uint32_t>({0}));
  dv_p->lowering().addPipelineIndexTensor(stashIndex);

  // Create the stash size tensor.
  const auto stashSizeTensor =
      getConst(poplar::UNSIGNED_INT, {}, stashSize, "stash_size");

  // Grow program to take slice of stash at the stash index.
  poplar::Tensor actFromStash;

  if (canDynamicSliceRestore) {
    actFromStash = growDynamicSliceRestore(prog, stashIndex, stash);
  } else {
    actFromStash = growStaticSliceRestore(prog, stashSize, stashIndex, stash);
  }

  // Create a "1" tensor and grow program to increment stash index by 1.
  auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one");
  popops::addInPlace(graph(), stashIndex, one, prog, debugContext());
  popops::remInPlace(
      graph(), stashIndex, stashSizeTensor, prog, debugContext());

  return actFromStash;
}

template <typename Derived>
poplar::Tensor RestoreBaseOpx<Derived>::growStaticSliceRestore(
    poplar::program::Sequence &prog,
    const int64_t stashSize,
    const poplar::Tensor &stashIndex,
    const poplar::Tensor &stash) const {

  // stash is (N, a, b, c). Output is (a, b, c) at index stashIndex.

  // Creates (1, a, b, c) tensor.
  auto actFromStash = popops::createSliceTensor(
      graph(), stash, {0}, {1}, 1, debugContext("static-restore/out-slice"));

  poplar::program::Switch switchCase(
      stashIndex.reshape({}),
      debugContext("static-restore/switch-on-stash-index"));

  for (int64_t i = 0; i < stashSize; i++) {
    const auto inSliceAtIdx = stash.slice(i, i + 1, 0);
    switchCase.add(
        i,
        poplar::program::Copy(
            inSliceAtIdx,
            actFromStash,
            false,
            debugContext("static-restore/switch-copy-" + std::to_string(i))));
  }

  prog.add(switchCase);

  return actFromStash.squeeze({0});
}

template <typename Derived>
poplar::Tensor RestoreBaseOpx<Derived>::growDynamicSliceRestore(
    poplar::program::Sequence &prog,
    const poplar::Tensor &stashIndex,
    const poplar::Tensor &stash) const {

  auto actFromStash =
      popops::dynamicSlice(graph(),
                           stash,
                           stashIndex,
                           {0},
                           {1},
                           prog,
                           debugContext("grow_restore_dynamic_slice"));

  return actFromStash.squeeze({0});
}

void RestoreInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto actToRestore = getInTensor(RestoreInplaceOp::getActToRestoreInIndex());
  auto stash        = getInTensor(RestoreInplaceOp::getStashInIndex());

  const auto actFromStash = growRestore(prog, stash);

  prog.add(
      poplar::program::Copy(actFromStash, actToRestore, false, debugContext()));
  setOutTensor(RestoreInplaceOp::getRestoredActOutIndex(), actToRestore);
}

RestoreInplaceOpx::RestoreInplaceOpx(Op *op, Devicex *devicex)
    : RestoreBaseOpx(op, devicex) {}

void RestoreOpx::grow(poplar::program::Sequence &prog) const {
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
