#include <popnn/NonLinearity.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/popx/op/tanhx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

TanhOpx::TanhOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TanhOp>(op, {Onnx::Operators::Tanh_6});
}

void TanhOpx::grow(poplar::program::Sequence &prog) const {
  auto in_tensor  = get(inId(TanhOp::getInIndex()));
  auto out_id     = outId(TanhOp::getOutIndex());
  auto out_tensor = popnn::nonLinearity(
      graph(), popnn::NonLinearityType::TANH, in_tensor, prog, out_id);
  insert(out_id, out_tensor);
}

InputCreatorType TanhOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor TanhOpx::unwindTensorLayout(poplar::Tensor tensor) const {
  return tensor;
}

TanhGradOpx::TanhGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TanhGradOp>(op, Onnx::GradOperators::TanhGrad);
}

void TanhGradOpx::grow(poplar::program::Sequence &prog) const {
  auto fwd_out    = get(inId(TanhGradOp::getFwdOutInIndex()));
  auto grad_out   = get(inId(TanhGradOp::getGradInIndex()));
  auto out_id     = outId(TanhGradOp::getOutIndex());
  auto out_tensor = popnn::nonLinearityInputGradient(
      graph(), popnn::NonLinearityType::TANH, fwd_out, grad_out, prog, out_id);
  insert(out_id, out_tensor);
}

namespace {
OpxCreator<TanhOpx> tanhOpxCreator(Onnx::Operators::Tanh_6);
OpxCreator<TanhGradOpx> tanhGradOpxCreator(Onnx::GradOperators::TanhGrad);
} // namespace

} // namespace popx
} // namespace poponnx
