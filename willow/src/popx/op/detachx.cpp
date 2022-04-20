// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popart/op/detach.hpp>
#include <popart/popx/op/detachx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

DetachOpx::DetachOpx(popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DetachOp>(op, Onnx::CustomOperators::Detach_1);
}

void DetachOpx::grow(snap::program::Sequence &prog) const {
  auto input = getInTensor(DetachOp::getInIndex());

  auto output = cloneNcopy(prog, input);
  setOutTensor(DetachOp::getOutIndex(), output);
}

DetachInplaceOpx::DetachInplaceOpx(Op *op, Devicex *devicex)
    : popart::popx::ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DetachInplaceOp>(op);
}

void DetachInplaceOpx::grow(snap::program::Sequence &) const {
  setOutTensor(DetachOp::getOutIndex(), getInTensor(DetachOp::getInIndex()));
}

namespace {
OpxCreator<DetachOpx> detachOpxCreator(Onnx::CustomOperators::Detach_1);
OpxCreator<DetachInplaceOpx>
    detachInplaceOpxCreator(Onnx::CustomOperators::DetachInplace);
} // namespace

} // namespace popx
} // namespace popart
