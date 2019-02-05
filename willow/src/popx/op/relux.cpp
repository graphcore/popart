#include <poponnx/device.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/relu.hpp>
#include <poponnx/popx/op/relux.hpp>
#include <poponnx/popx/opxmanager.hpp>
#include <poponnx/tensorindex.hpp>

#include <popnn/NonLinearity.hpp>

namespace poponnx {
namespace popx {

ReluOpx::ReluOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReluOp>(op, Onnx::Operators::Relu_6);
}

ReluInplaceOpx::ReluInplaceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReluInplaceOp>(op, Onnx::CustomOperators::ReluInplace);
}

void ReluOpx::grow(poplar::program::Sequence &prog) const {

  // There is only an in-place poplibs Relu. We therefore clone first,
  auto outTensor = cloneNcopy(prog, inId(0));

  // and apply the inplace relu.
  popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::RELU, outTensor, prog, outId(0));

  insert(outId(0), outTensor);
}

InputCreatorType ReluOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor
ReluOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

void ReluInplaceOpx::grow(poplar::program::Sequence &prog) const {
  // apply the inplace relu,
  popnn::nonLinearityInPlace(
      graph(), popnn::NonLinearityType::RELU, get(inId(0)), prog, inId(0));

  insert(outId(0), get(inId(0)));
}

InputCreatorType ReluInplaceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor ReluInplaceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                  InIndex,
                                                  OutIndex) const {
  return tensor;
}

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReluGradOp>(op, Onnx::GradOperators::ReluGrad);
}

void ReluGradOpx::grow(poplar::program::Sequence &prog) const {

  ReluGradOp &rgop = getOp<ReluGradOp>();

  auto outTensor = popnn::nonLinearityInputGradient(
      graph(),                               // graph,
      popnn::NonLinearityType::RELU,         // nonLinearityType,
      get(inId(rgop.getReludInIndex())),     //  out,
      get(inId(rgop.getGradReludInIndex())), //  outGradient,
      prog,                                  // prog,
      idStr()                                // debugPrefix
  );

  insert(op_p->output->id(0), outTensor);
}

namespace {
OpxCreator<ReluOpx> reluxOpxCreator(Onnx::Operators::Relu_6);
OpxCreator<ReluInplaceOpx>
    reluxInplaceOpxCreator(Onnx::CustomOperators::ReluInplace);
OpxCreator<ReluGradOpx> reluxGradOpxCreator(Onnx::GradOperators::ReluGrad);
} // namespace

} // namespace popx
} // namespace poponnx
