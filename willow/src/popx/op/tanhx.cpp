#include <popnn/NonLinearity.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/popx/op/tanhx.hpp>

namespace poponnx {
namespace popx {

TanhOpx::TanhOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<TanhOp>()) {
    throw error("cannot create TanhOpx from " + op->op_type());
  }
}

void TanhOpx::grow(poplar::program::Sequence &prog) const {
  auto in_tensor  = get(inId(TanhOp::getInIndex()));
  auto out_id     = outId(TanhOp::getOutIndex());
  auto out_tensor = popnn::nonLinearity(
      graph(), popnn::NonLinearityType::TANH, in_tensor, prog, out_id);
  insert(out_id, out_tensor);
}

TanhGradOpx::TanhGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<TanhGradOp>()) {
    throw error("cannot create TanhGradOpx from " + op->op_type());
  }
}

void TanhGradOpx::grow(poplar::program::Sequence &prog) const {
  auto fwd_out    = get(inId(TanhGradOp::getFwdOutInIndex()));
  auto grad_out   = get(inId(TanhGradOp::getGradInIndex()));
  auto out_id     = outId(TanhGradOp::getOutIndex());
  auto out_tensor = popnn::nonLinearityInputGradient(
      graph(), popnn::NonLinearityType::TANH, fwd_out, grad_out, prog, out_id);
  insert(out_id, out_tensor);
}

} // namespace popx
} // namespace poponnx
