// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popops/SequenceSlice.hpp>
#include <popart/op/sequenceslice.hpp>
#include <popart/popx/op/sequenceslicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

namespace {

void growSequenceSlice(const PopOpx *opx,
                       snap::program::Sequence &prog,
                       bool inplace) {
  auto source =
      opx->getInTensor(SequenceSliceOp::getSourceInIndex()).getPoplarTensor();
  auto destination = opx->getInTensor(SequenceSliceOp::getDestinationInIndex());
  auto N = opx->getInTensor(SequenceSliceOp::getNInIndex()).getPoplarTensor();
  auto sourceOffset =
      opx->getInTensor(SequenceSliceOp::getSourceOffsetInIndex())
          .getPoplarTensor();
  auto destOffset = opx->getInTensor(SequenceSliceOp::getDestOffsetInIndex())
                        .getPoplarTensor();

  if (!inplace) {
    destination = opx->cloneNcopy(prog, destination);
  }

  popops::sequenceSlice(opx->graph().getPoplarGraph(),
                        source,
                        destination.getPoplarTensor(),
                        N,
                        sourceOffset,
                        destOffset,
                        opx->getOp<SequenceSliceOp>().zeroUnused,
                        prog.getPoplarSequence(),
                        opx->debugContext());

  opx->setOutTensor(SequenceSliceOp::getOutIndex(), destination);
}

} // namespace

SequenceSliceOpx::SequenceSliceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SequenceSliceOp>(op);
}

void SequenceSliceOpx::grow(snap::program::Sequence &prog) const {
  growSequenceSlice(this, prog, false);
}

SequenceSliceInplaceOpx::SequenceSliceInplaceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SequenceSliceInplaceOp>(op);
}

void SequenceSliceInplaceOpx::grow(snap::program::Sequence &prog) const {
  growSequenceSlice(this, prog, true);
}

namespace {
// Ops
OpxCreator<SequenceSliceOpx>
    sequenceSliceOpxCreator(Onnx::CustomOperators::SequenceSlice_1);
OpxCreator<SequenceSliceInplaceOpx>
    sequenceSliceInplaceOpxCreator(Onnx::CustomOperators::SequenceSliceInplace);
} // namespace

} // namespace popx
} // namespace popart
