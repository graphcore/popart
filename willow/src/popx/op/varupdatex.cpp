#include <poponnx/error.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/popx/op/varupdatex.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

SGDVarUpdateOpx::SGDVarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SGDVarUpdateOp>(op, Onnx::CustomOperators::SgdVarUpdate);
}

void SGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // Weight update (matching pytorch implementation):
  //   w <- w * (1 - lr * wd) - lr * delta

  // First update weights with weight decay
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::_2),
                     {getInTensor(SGDVarUpdateOp::getVarInIndex()),
                      getInTensor(SGDVarUpdateOp::getWeightDecayInIndex())},
                     prog,
                     idStr());

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      getInTensor(SGDVarUpdateOp::getVarInIndex()),     // weights
      getInTensor(SGDVarUpdateOp::getVarGradInIndex()), // weightDeltas
      getInTensor(SGDVarUpdateOp::getLearnRateInIndex()),
      prog,
      idStr());

  // no poplar::Tensors to insert
}

ConstSGDVarUpdateOpx::ConstSGDVarUpdateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ConstSGDVarUpdateOp>(op, Onnx::CustomOperators::ConstSgdVarUpdate);
}

void ConstSGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getOp<ConstSGDVarUpdateOp>();

  // Weight update (matching pytorch implementation):
  //   w <- w * (1 - lr * wd) - lr * delta

  // First update weights with weight decay (only if user has
  // specified non-zero weight decay)
  if (vu_op.getWeightDecay() != 0.0f) {
    float weightDecayScaleFactor =
        1 - (vu_op.getWeightDecay() * vu_op.getLearnRate());

    popops::mapInPlace(graph(),
                       pe::Mul(pe::_1, pe::Const(weightDecayScaleFactor)),
                       {getInTensor(SGDVarUpdateOp::getVarInIndex())},
                       prog,
                       idStr());
  }

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      getInTensor(vu_op.getVarInIndex()),     // weights
      getInTensor(vu_op.getVarGradInIndex()), // weightDeltas
      vu_op.getLearnRate(),
      prog,
      idStr());

  // no poplar::Tensors to insert
}

CopyVarUpdateOpx::CopyVarUpdateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<CopyVarUpdateOp>(op, Onnx::CustomOperators::CopyVarUpdate);
}

void CopyVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getOp<CopyVarUpdateOp>();
  poplar::program::Copy copy(getInTensor(CopyVarUpdateOp::getVarFromInIndex()),
                             getInTensor(CopyVarUpdateOp::getVarToInIndex()));
  prog.add(copy);

  // no poplar::Tensors to insert
}

namespace {
OpxCreator<SGDVarUpdateOpx>
    sgdVarUpdateOpxCreator(Onnx::CustomOperators::SgdVarUpdate);
OpxCreator<ConstSGDVarUpdateOpx>
    constSgdVarUpdateOpxCreator(Onnx::CustomOperators::ConstSgdVarUpdate);
OpxCreator<CopyVarUpdateOpx>
    copyVarUpdateOpxCreator(Onnx::CustomOperators::CopyVarUpdate);
} // namespace

} // namespace popx
} // namespace poponnx
