#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/flatten.hpp>
#include <poponnx/popx/op/flattenx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

FlattenAliasOpx::FlattenAliasOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<FlattenAliasOp>(op);
}

poplar::Tensor FlattenAliasOpx::grow(poplar::program::Sequence &prog,
                                     poplar::Tensor &input) const {
  auto input_copy = cloneNcopy(prog, input);

  return input_copy.reshape(outInfo(FlattenOp::getOutIndex()).shape_szt());
}

void FlattenAliasOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(FlattenOp::getInIndex());

  setOutTensor(FlattenOp::getOutIndex(), grow(prog, input));
}

void FlattenOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(FlattenOp::getInIndex());

  auto output = cloneNcopy(prog, FlattenAliasOpx::grow(prog, input));

  setOutTensor(FlattenOp::getOutIndex(), output);
}

FlattenGradOpx::FlattenGradOpx(Op *op, Devicex *devicex)
    : ReshapeOpx(op, devicex) {
  verifyOp<FlattenGradOp>(op, Onnx::GradOperators::FlattenGrad);
}

namespace {
OpxCreator<FlattenOpx> flattenOpxCreator({Onnx::Operators::Flatten_1,
                                          Onnx::Operators::Flatten_9});
OpxCreator<FlattenAliasOpx>
    flattenAliasOpxCreator(Onnx::CustomOperators::FlattenAlias);
OpxCreator<FlattenGradOpx>
    reshapeGradOpxCreator(Onnx::GradOperators::FlattenGrad);
} // namespace

} // namespace popx
} // namespace poponnx
