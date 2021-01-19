// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/reverse.hpp>
#include <popart/popx/op/reversex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ReverseBaseOpx::ReverseBaseOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReverseBaseOp>(op);
}

ReverseOpx::ReverseOpx(Op *op, Devicex *devicex) : ReverseBaseOpx(op, devicex) {
  verifyOp<ReverseOp>(op);
}

void ReverseOpx::grow(poplar::program::Sequence &prog) const {
  auto t = getInTensor(ReverseOp::getInIndex());
  for (auto dim : dynamic_cast<ReverseOp *>(op_p)->getDimensions()) {
    t = t.reverse(static_cast<unsigned>(dim));
  }

  setOutTensor(ReverseOp::getOutIndex(), cloneNcopy(prog, t));
}

poplar::Tensor ReverseBaseOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                  InIndex,
                                                  OutIndex) const {
  for (auto dim : dynamic_cast<ReverseOp *>(op_p)->getDimensions()) {
    tensor = tensor.reverse(static_cast<unsigned>(dim));
  }

  return tensor;
}

view::RegMap ReverseBaseOpx::unwindRegion(InIndex inIndex,
                                          OutIndex outIndex) const {
  ReverseBaseOp *op = dynamic_cast<ReverseBaseOp *>(this->op_p);
  return op->bwdRegMap(inIndex, outIndex);
}

ReverseInplaceOpx::ReverseInplaceOpx(Op *op, Devicex *devicex)
    : ReverseBaseOpx(op, devicex) {
  verifyOp<ReverseInplaceOp>(op);
}

void ReverseInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(ReverseOp::getInIndex());
  for (auto dim : dynamic_cast<ReverseInplaceOp *>(op_p)->getDimensions()) {
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
