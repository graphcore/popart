#include <algorithm>
#include <popart/op/slice.hpp>
#include <popart/op/slicegrad.hpp>
#include <popart/popx/op/slicegradx.hpp>
#include <popart/popx/op/slicex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SliceOpx::SliceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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

void SliceInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(SliceOp::getInIndex());
  for (auto slice : getSliceInplaceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }
  setOutTensor(SliceOp::getOutIndex(), t);
}

SliceOp *SliceOpx::getSliceOp() const { return dynamic_cast<SliceOp *>(op_p); }

SliceInplaceOp *SliceInplaceOpx::getSliceInplaceOp() const {
  return dynamic_cast<SliceInplaceOp *>(op_p);
}

SliceGradOpx::SliceGradOpx(Op *op, Devicex *devicex) : PadOpx(op, devicex) {
  verifyOp<SliceGradOp>(op, Onnx::GradOperators::SliceGrad);
}

SliceInplaceOpx::SliceInplaceOpx(Op *op_, Devicex *devicex)
    : Opx(op_, devicex) {
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
