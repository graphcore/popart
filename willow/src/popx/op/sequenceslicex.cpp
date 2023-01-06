// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poplar/Tensor.hpp>
#include <popops/SequenceSlice.hpp>
#include <popart/op/sequenceslice.hpp>
#include <popart/popx/op/sequenceslicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

namespace {

void growSequenceSlice(const Opx *opx,
                       poplar::program::Sequence &prog,
                       bool inplace) {
  auto source      = opx->getInTensor(SequenceSliceOp::getSourceInIndex());
  auto destination = opx->getInTensor(SequenceSliceOp::getDestinationInIndex());
  auto N           = opx->getInTensor(SequenceSliceOp::getNInIndex());
  auto sourceOffset =
      opx->getInTensor(SequenceSliceOp::getSourceOffsetInIndex());
  auto destOffset = opx->getInTensor(SequenceSliceOp::getDestOffsetInIndex());

  if (!inplace) {
    destination = opx->cloneNcopy(prog, destination);
  }

  popops::sequenceSlice(opx->graph(),
                        source,
                        destination,
                        N,
                        sourceOffset,
                        destOffset,
                        opx->getOp<SequenceSliceOp>().zeroUnused,
                        prog,
                        opx->debugContext());

  opx->setOutTensor(SequenceSliceOp::getOutIndex(), destination);
}

} // namespace

SequenceSliceOpx::SequenceSliceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SequenceSliceOp>(op);
}

void SequenceSliceOpx::grow(poplar::program::Sequence &prog) const {
  growSequenceSlice(this, prog, false);
}

SequenceSliceInplaceOpx::SequenceSliceInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SequenceSliceInplaceOp>(op);
}

void SequenceSliceInplaceOpx::grow(poplar::program::Sequence &prog) const {
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
