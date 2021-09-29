// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/reverse.hpp>
#include <popart/popx/op/reversex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ReverseBaseOpx::ReverseBaseOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ReverseBaseOp>(op);
}

ReverseOpx::ReverseOpx(Op *op, Devicex *devicex) : ReverseBaseOpx(op, devicex) {
  verifyOp<ReverseOp>(op);
}

void ReverseOpx::grow(snap::program::Sequence &prog) const {
  auto t = getInTensor(ReverseOp::getInIndex()).getPoplarTensor();
  for (auto dim : getOp<ReverseOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  setOutTensor(ReverseOp::getOutIndex(),
               cloneNcopy(prog, snap::Tensor{t, graph()}));
}

snap::Tensor ReverseBaseOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  poplar::Tensor t = tensor.getPoplarTensor();
  for (auto dim : getOp<ReverseBaseOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  return snap::Tensor{t, graph()};
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

void ReverseInplaceOpx::grow(snap::program::Sequence &) const {
  auto t = getInTensor(ReverseOp::getInIndex()).getPoplarTensor();
  for (auto dim : getOp<ReverseInplaceOp>().getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  setOutTensor(ReverseOp::getOutIndex(), snap::Tensor{t, graph()});
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
