// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/slice.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/slicex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <popart/ir.hpp>
#include <popart/tensors.hpp>

#include <popops/Pad.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

BaseSliceOpx::BaseSliceOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

InputCreatorType BaseSliceOpx::getInputCreatorType(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("inIndex should be 0 in BaseSliceOpx");
  }
  return InputCreatorType::CanUnwind;
}

snap::Tensor
BaseSliceOpx::unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

view::RegMap BaseSliceOpx::unwindRegion(InIndex inIndex,
                                        OutIndex outIndex) const {
  BaseSliceOp *op = dynamic_cast<BaseSliceOp *>(this->op_p);
  return op->bwdRegMap(inIndex, outIndex);
}

SliceOpx::SliceOpx(Op *op, Devicex *devicex) : BaseSliceOpx(op, devicex) {
  verifyOp<SliceOp>(op);
}

void SliceOpx::grow(poplar::program::Sequence &prog) const {
  auto t = getInTensor(SliceOp::getInIndex()).getPoplarTensor();
  for (auto slice : getSliceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
    if (slice.flip) {
      t = t.reverse(static_cast<unsigned>(slice.axis));
    }
  }
  // we clone and copy t, as this in not an inplace op
  setOutTensor(SliceOp::getOutIndex(),
               cloneNcopy(prog, snap::Tensor{t, graph()}));
}

SliceOp *SliceOpx::getSliceOp() const { return dynamic_cast<SliceOp *>(op_p); }

void SliceInplaceOpx::grow(poplar::program::Sequence &) const {
  auto t = getInTensor(SliceOp::getInIndex()).getPoplarTensor();
  for (auto slice : getSliceInplaceOp()->getSlices()) {
    t = t.slice(slice.start, slice.end, static_cast<unsigned>(slice.axis));
    if (slice.flip) {
      t = t.reverse(static_cast<unsigned>(slice.axis));
    }
  }
  setOutTensor(SliceOp::getOutIndex(), snap::Tensor{t, graph()});
}

SliceInplaceOp *SliceInplaceOpx::getSliceInplaceOp() const {
  return dynamic_cast<SliceInplaceOp *>(op_p);
}

SliceInplaceOpx::SliceInplaceOpx(Op *op_, Devicex *devicex)
    : BaseSliceOpx(op_, devicex) {
  verifyOp<SliceInplaceOp>(op_);
}

namespace {
OpxCreator<SliceOpx> sliceOpxCreator({Onnx::Operators::Slice_1,
                                      Onnx::Operators::Slice_10,
                                      Onnx::Operators::Slice_11});
OpxCreator<SliceInplaceOpx>
    sliceInplaceOpxCreator(Onnx::CustomOperators::SliceInplace);
} // namespace

} // namespace popx
} // namespace popart
