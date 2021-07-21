// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/op/restore.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/restorex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

template <typename Derived>
RestoreBaseOpx<Derived>::RestoreBaseOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<typename Derived::OpType>(op);

  // Note RestoreInplaceOp derives RestoreOp.
  canDynamicSliceRestore =
      !(inInfo(RestoreOp::getStashInIndex()).dataType() == DataType::INT8 ||
        inInfo(RestoreOp::getStashInIndex()).dataType() == DataType::UINT8);
}

template <typename Derived>
snap::Tensor
RestoreBaseOpx<Derived>::growRestore(poplar::program::Sequence &prog,
                                     const snap::Tensor &stash) const {
  const auto &op       = getOp<typename Derived::OpType>();
  const auto stashSize = op.getStashSize();

  // Create the stash index tensor.
  const auto stashIndex = graph().getPoplarGraph().addVariable(
      poplar::UNSIGNED_INT, {1}, debugContext());
  graph().getPoplarGraph().setTileMapping(stashIndex, 0);
  graph().getPoplarGraph().setInitialValue(stashIndex,
                                           poplar::ArrayRef<uint32_t>({0}));

  // Create the stash size tensor.
  const auto stashSizeTensor =
      getConst(poplar::UNSIGNED_INT, {}, stashSize, "stash_size")
          .getPoplarTensor();

  // Grow program to take slice of stash at the stash index.
  snap::Tensor actFromStash;

  if (canDynamicSliceRestore) {
    actFromStash =
        growDynamicSliceRestore(prog, snap::Tensor{stashIndex, graph()}, stash);
  } else {
    actFromStash = growStaticSliceRestore(
        prog, stashSize, snap::Tensor{stashIndex, graph()}, stash);
  }

  // Create a "1" tensor and grow program to increment stash index by 1.
  auto one = getConst(poplar::UNSIGNED_INT, {}, 1.0, "one").getPoplarTensor();
  popops::addInPlace(
      graph().getPoplarGraph(), stashIndex, one, prog, debugContext());
  popops::remInPlace(graph().getPoplarGraph(),
                     stashIndex,
                     stashSizeTensor,
                     prog,
                     debugContext());

  return actFromStash;
}

template <typename Derived>
snap::Tensor RestoreBaseOpx<Derived>::growStaticSliceRestore(
    poplar::program::Sequence &prog,
    const int64_t stashSize,
    const snap::Tensor &stashIndex,
    const snap::Tensor &stash) const {

  // stash is (N, a, b, c). Output is (a, b, c) at index stashIndex.

  // Creates (1, a, b, c) tensor.
  poplar::Tensor actFromStash =
      popops::createSliceTensor(graph().getPoplarGraph(),
                                stash.getPoplarTensor(),
                                {0},
                                {1},
                                1,
                                debugContext("static-restore/out-slice"));

  poplar::program::Switch switchCase(
      stashIndex.reshape({}).getPoplarTensor(),
      debugContext("static-restore/switch-on-stash-index"));

  for (int64_t i = 0; i < stashSize; i++) {
    const auto inSliceAtIdx = stash.getPoplarTensor().slice(i, i + 1, 0);
    switchCase.add(
        i,
        poplar::program::Copy(
            inSliceAtIdx,
            actFromStash,
            false,
            debugContext("static-restore/switch-copy-" + std::to_string(i))));
  }

  prog.add(switchCase);

  return snap::Tensor{actFromStash.squeeze({0}), graph()};
}

template <typename Derived>
snap::Tensor RestoreBaseOpx<Derived>::growDynamicSliceRestore(
    poplar::program::Sequence &prog,
    const snap::Tensor &stashIndex,
    const snap::Tensor &stash) const {

  auto actFromStash =
      popops::dynamicSlice(graph().getPoplarGraph(),
                           stash.getPoplarTensor(),
                           stashIndex.getPoplarTensor(),
                           {0},
                           {1},
                           prog,
                           debugContext("grow_restore_dynamic_slice"));

  return snap::Tensor{actFromStash.squeeze({0}), graph()};
}

void RestoreInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto actToRestore = getInTensor(RestoreInplaceOp::getActToRestoreInIndex());
  auto stash        = getInTensor(RestoreInplaceOp::getStashInIndex());

  const auto actFromStash = growRestore(prog, stash);

  prog.add(poplar::program::Copy(actFromStash.getPoplarTensor(),
                                 actToRestore.getPoplarTensor(),
                                 false,
                                 debugContext()));
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
