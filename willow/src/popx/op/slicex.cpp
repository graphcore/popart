#include <algorithm>
#include <poponnx/op/slice.hpp>
#include <poponnx/op/slicegrad.hpp>
#include <poponnx/popx/op/slicegradx.hpp>
#include <poponnx/popx/op/slicex.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

SliceOpx::SliceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SliceOp>(op);
}

void SliceOpx::grow(poplar::program::Sequence &) const {
  logging::devicex::trace("SliceOpx::grow");
  auto t = get(inId(SliceOp::getInIndex()));

  for (auto slice : getSliceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
  }

  insert(outId(SliceOp::getOutIndex()), t);
}

SliceOp *SliceOpx::getSliceOp() const { return dynamic_cast<SliceOp *>(op_p); }

SliceGradOpx::SliceGradOpx(Op *op, Devicex *devicex) : PadOpx(op, devicex) {
  verifyOp<SliceGradOp>(op, Onnx::GradOperators::SliceGrad);
}

namespace {
OpxCreator<SliceOpx> sliceOpxCreator(Onnx::Operators::Slice_1);
OpxCreator<SliceGradOpx> sliceGradOpxCreator(Onnx::GradOperators::SliceGrad);
} // namespace

} // namespace popx
} // namespace poponnx
