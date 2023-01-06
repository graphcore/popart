// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/reverse.hpp>
#include <popart/popx/op/reversex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/popx/opx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

ReverseBaseOpx::ReverseBaseOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReverseBaseOp>(op);
}

ReverseOpx::ReverseOpx(Op *op, Devicex *devicex) : ReverseBaseOpx(op, devicex) {
  verifyOp<ReverseOp>(op);
}

void ReverseOpx::grow(poplar::program::Sequence &prog) const {
  auto t = getInTensor(ReverseOp::getInIndex());
  for (auto dim : getOp<ReverseOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  setOutTensor(ReverseOp::getOutIndex(), cloneNcopy(prog, t));
}

poplar::Tensor ReverseBaseOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                  InIndex,
                                                  OutIndex) const {
  auto t = tensor;
  for (auto dim : getOp<ReverseBaseOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  return t;
}

view::RegMap ReverseBaseOpx::unwindRegion(InIndex inIndex,
                                          OutIndex outIndex) const {
  ReverseBaseOp &op = getOp<ReverseBaseOp>();
  return op.bwdRegMap(inIndex, outIndex);
}

ReverseInplaceOpx::ReverseInplaceOpx(Op *op, Devicex *devicex)
    : ReverseBaseOpx(op, devicex) {
  verifyOp<ReverseInplaceOp>(op);
}

void ReverseInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(ReverseOp::getInIndex());
  for (auto dim : getOp<ReverseInplaceOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  setOutTensor(ReverseOp::getOutIndex(), t);
}

ReverseGradOpx::ReverseGradOpx(Op *op, Devicex *devicex)
    : ReverseOpx(op, devicex) {
  verifyOp<ReverseGradOp>(op, Onnx::GradOperators::ReverseGrad);
}

namespace {
OpxCreator<ReverseOpx> reverseOpxCreator({Onnx::AiGraphcore::OpSet1::Reverse});
OpxCreator<ReverseInplaceOpx>
    reverseInplaceOpxCreator(Onnx::CustomOperators::ReverseInplace);
OpxCreator<ReverseGradOpx>
    reverseGradOpxCreator(Onnx::GradOperators::ReverseGrad);
} // namespace

} // namespace popx
} // namespace popart
