// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/reshape.hpp>
#include <popart/popx/op/reshapex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace popx {

// Test note : scale by 1.0001 in grad op makes the test fail. Good.
void ReshapeOpx::grow(poplar::program::Sequence &prog) const {
  // not in-place, so cloning input
  auto outTensor = cloneNcopy(prog, getInTensor(ReshapeOp::getInIndex()));
  outTensor = outTensor.reshape(outInfo(ReshapeOp::getOutIndex()).shape_szt());
  setOutTensor(ReshapeOp::getOutIndex(), outTensor);
}

void ReshapeInplaceOpx::grow(poplar::program::Sequence &) const {
  auto outTensor = getInTensor(ReshapeOp::getInIndex());
  outTensor = outTensor.reshape(outInfo(ReshapeOp::getOutIndex()).shape_szt());
  setOutTensor(ReshapeOp::getOutIndex(), outTensor);
}

ReshapeBaseOpx::ReshapeBaseOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ReshapeBaseOp>(op);
}

ReshapeOpx::ReshapeOpx(Op *op, Devicex *devicex) : ReshapeBaseOpx(op, devicex) {
  verifyOp<ReshapeOp>(op);
}

ReshapeInplaceOpx::ReshapeInplaceOpx(Op *op, Devicex *devicex)
    : ReshapeBaseOpx(op, devicex) {
  verifyOp<ReshapeInplaceOp>(op);
}

InputCreatorType ReshapeBaseOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor ReshapeBaseOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  return snap::Tensor{tensor.getPoplarTensor().reshape(
                          inInfo(ReshapeOp::getInIndex()).shape_szt()),
                      graph()};
}

view::RegMap ReshapeBaseOpx::unwindRegion(InIndex inIndex,
                                          OutIndex outIndex) const {
  ReshapeBaseOp *op = dynamic_cast<ReshapeBaseOp *>(this->op_p);
  return op->bwdRegMap(inIndex, outIndex);
}

ReshapeGradOpx::ReshapeGradOpx(Op *op, Devicex *devicex)
    : ReshapeOpx(op, devicex) {
  verifyOp<ReshapeGradOp>(op, Onnx::GradOperators::ReshapeGrad);
}

namespace {
OpxCreator<ReshapeOpx> reshapeOpxCreator(Onnx::Operators::Reshape_5);
OpxCreator<ReshapeInplaceOpx>
    reshapeInplaceOpxCreator(Onnx::CustomOperators::ReshapeInplace);
OpxCreator<ReshapeGradOpx>
    reshapeGradOpxCreator(Onnx::GradOperators::ReshapeGrad);
} // namespace

} // namespace popx
} // namespace popart
