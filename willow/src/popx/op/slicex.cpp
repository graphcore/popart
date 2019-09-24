#include <algorithm>

#include <popart/op.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/slicegrad.hpp>
#include <popart/popx/op/slicegradx.hpp>
#include <popart/popx/op/slicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <popart/ir.hpp>
#include <popart/tensors.hpp>

namespace popart {
namespace popx {

BaseSliceOpx::BaseSliceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

InputCreatorType BaseSliceOpx::getInputCreatorType(InIndex inIndex) const {
  if (!dynamic_cast<BaseSliceOp *>(op_p)->allSlices.empty()) {
    return InputCreatorType::CANUNWIND_MULTIPLE_CREATORS;
  } else {
    return Opx::getInputCreatorType(inIndex);
  }
}
poplar::Tensor
BaseSliceOpx::unwindTensorLayout(std::vector<poplar::Tensor> tensors,
                                 InIndex,
                                 OutIndex) const {

  BaseSliceOp *op = dynamic_cast<BaseSliceOp *>(this->op_p);
  logging::opx::debug("BaseSliceOpx::unwindTensorLayout dim:{} inputs:{}",
                      op->unwindConcatDim,
                      tensors.size());
  return poplar::concat(tensors, op->unwindConcatDim);
}

std::vector<Op *> BaseSliceOpx::getCreatorCandicates(InIndex) const {

  std::vector<Op *> creators;

  BaseSliceOp *op = dynamic_cast<BaseSliceOp *>(this->op_p);

  // Get the list of op's the consume the all the slices of the input tensor
  for (auto t : op->allSlices) {
    auto consumerOps = op->getIr().getTensor(t)->consumers.getOps();
    creators.insert(creators.end(), consumerOps.begin(), consumerOps.end());
  }

  return creators;
}

SliceOpx::SliceOpx(Op *op, Devicex *devicex) : BaseSliceOpx(op, devicex) {
  verifyOp<SliceOp>(op);
}

void SliceOpx::grow(poplar::program::Sequence &prog) const {
  auto t = getInTensor(SliceOp::getInIndex());
  for (auto slice : getSliceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }
  // we clone and copy t, as this in not an inplace op
  setOutTensor(SliceOp::getOutIndex(), cloneNcopy(prog, t));
}

SliceOp *SliceOpx::getSliceOp() const { return dynamic_cast<SliceOp *>(op_p); }

void SliceInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(SliceOp::getInIndex());
  for (auto slice : getSliceInplaceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }
  setOutTensor(SliceOp::getOutIndex(), t);
}

SliceInplaceOp *SliceInplaceOpx::getSliceInplaceOp() const {
  return dynamic_cast<SliceInplaceOp *>(op_p);
}

SliceGradOpx::SliceGradOpx(Op *op, Devicex *devicex) : PadOpx(op, devicex) {
  verifyOp<SliceGradOp>(op, Onnx::GradOperators::SliceGrad);
}

SliceInplaceOpx::SliceInplaceOpx(Op *op_, Devicex *devicex)
    : BaseSliceOpx(op_, devicex) {
  verifyOp<SliceInplaceOp>(op_);
}

namespace {
OpxCreator<SliceOpx> sliceOpxCreator({Onnx::Operators::Slice_1,
                                      Onnx::Operators::Slice_10});
OpxCreator<SliceInplaceOpx>
    sliceInplaceOpxCreator(Onnx::CustomOperators::SliceInplace);
OpxCreator<SliceGradOpx> sliceGradOpxCreator(Onnx::GradOperators::SliceGrad);
} // namespace

} // namespace popx
} // namespace popart
