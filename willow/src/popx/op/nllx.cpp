#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/popx/op/nllx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include "popops/Encoding.hpp"
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

NllOpx::NllOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<NllOp>(op, Onnx::CustomOperators::Nll);
}

void NllOpx::grow(poplar::program::Sequence &prog) const {
  TensorId labelId     = getOp<NllOp>().nlll()->labelTensorId();
  TensorId probsId     = getOp<NllOp>().nlll()->probsTensorId();
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
  verifyOp<NllGradOp>(op, Onnx::CustomGradOperators::NllGrad);
}

namespace {
static OpxCreator<NllOpx> nllOpxCreator(Onnx::CustomOperators::Nll);
static OpxCreator<NllGradOpx>
    nllGradOpxCreator(Onnx::CustomGradOperators::NllGrad);
} // namespace

} // namespace popx
} // namespace poponnx
