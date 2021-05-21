
#include <popart/names.hpp>
#include <popart/op/detach.hpp>
#include <popart/popx/op/detachx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {

namespace popx {

DetachOpx::DetachOpx(popart::Op *op, popart::popx::Devicex *devicex)
    : popart::popx::ElementWiseUnaryOpx(op, devicex) {
  verifyOp<DetachOp>(op, Onnx::CustomOperators::Detach_1);
}

void DetachOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(DetachOp::getInIndex());

  auto output = cloneNcopy(prog, input);
  setOutTensor(DetachOp::getOutIndex(), output);
}

DetachInplaceOpx::DetachInplaceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<DetachInplaceOp>(op);
}

void DetachInplaceOpx::grow(poplar::program::Sequence &) const {
  setOutTensor(DetachOp::getOutIndex(), getInTensor(DetachOp::getInIndex()));
}

namespace {
OpxCreator<DetachOpx> detachOpxCreator(Onnx::CustomOperators::Detach_1);
OpxCreator<DetachInplaceOpx>
    detachInplaceOpxCreator(Onnx::CustomOperators::DetachInplace);
} // namespace

} // namespace popx
} // namespace popart
