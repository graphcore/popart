#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/flatten.hpp>
#include <popart/popx/op/flattenx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

FlattenInplaceOpx::FlattenInplaceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex),
      outShape(op->outInfo(FlattenBaseOp::getOutIndex()).shape_szt()) {
  verifyOp<FlattenInplaceOp>(op);
}

FlattenOpx::FlattenOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex),
      outShape(op->outInfo(FlattenBaseOp::getOutIndex()).shape_szt()) {
  verifyOp<FlattenOp>(op);
}

void FlattenOpx::grow(poplar::program::Sequence &prog) const {
  auto input     = getInTensor(FlattenBaseOp::getInIndex());
  auto inputCopy = cloneNcopy(prog, input);
  setOutTensor(FlattenBaseOp::getOutIndex(), inputCopy.reshape(outShape));
}

void FlattenInplaceOpx::grow(poplar::program::Sequence &) const {
  auto input = getInTensor(FlattenBaseOp::getInIndex());
  setOutTensor(FlattenBaseOp::getOutIndex(), input.reshape(outShape));
}

FlattenGradOpx::FlattenGradOpx(Op *op, Devicex *devicex)
    : ReshapeOpx(op, devicex) {
  verifyOp<FlattenGradOp>(op, Onnx::GradOperators::FlattenGrad);
}

namespace {
OpxCreator<FlattenOpx> flattenOpxCreator({Onnx::Operators::Flatten_1,
                                          Onnx::Operators::Flatten_9,
                                          Onnx::Operators::Flatten_11});
OpxCreator<FlattenInplaceOpx>
    flattenInplaceOpxCreator(Onnx::CustomOperators::FlattenInplace);
OpxCreator<FlattenGradOpx>
    flattenGradOpxCreator(Onnx::GradOperators::FlattenGrad);
} // namespace

} // namespace popx
} // namespace popart
