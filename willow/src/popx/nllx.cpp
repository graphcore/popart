#include <poponnx/error.hpp>
#include <poponnx/nll.hpp>
#include <poponnx/popx/nllx.hpp>
#include <poponnx/util.hpp>

#include "popops/Encoding.hpp"
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace willow {
namespace popx {

NllOpx::NllOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::NLL) {
    throw error("cannot create NllOpx from " + op->op_type());
  }
}

NllOp *NllOpx::getNllOp() const { return dynamic_cast<NllOp *>(op_p); }

void NllOpx::grow(poplar::program::Sequence &prog) const {
  TensorId labelId     = getNllOp()->nlll()->labelTensorId();
  TensorId probsId     = getNllOp()->nlll()->probsTensorId();
  poplar::Tensor probs = get(probsId);

  // Tensor taking one-hot encoded output must be 2 dimensional
  // probs = probs.reshape({probs.dim(0), probs.numElements() / probs.dim(0)});
  auto oneHot = graph().clone(probs.elementType(), probs, "..OneHot");

  popops::encodeOneHot(graph(), get(labelId), oneHot, prog, "..Nll");

  // oneHot, from a tensor which is sparse with a single 1 per row,
  //           to a tensor which is sparse with a single p per row.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     oneHot,
                     probs,
                     prog,
                     "..mul");

  // sum rows, so that just the p corresponding to the label remains
  poplar::Tensor reduction =
      popops::reduce(graph(), oneHot, {1}, {popops::Operation::ADD}, prog);

  // and log it,
  popops::mapInPlace(
      graph(), popops::expr::UnaryOpType::LOGARITHM, reduction, prog, "..log");

  // and negate it.
  popops::mapInPlace(
      graph(), popops::expr::UnaryOpType::NEGATE, reduction, prog, "..neg");

  insert(outId(0), reduction);
}

NllGradOpx::NllGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::NLLGRAD) {
    throw error("cannot create NllGradOpx from " + op->op_type());
  }
}

NllGradOp *NllGradOpx::getNllGradOp() const {
  return dynamic_cast<NllGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
