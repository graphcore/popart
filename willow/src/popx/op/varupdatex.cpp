#include <poponnx/error.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/popx/op/varupdatex.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace poponnx {
namespace popx {

SGDVarUpdateOpx::SGDVarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SGDVarUpdateOp>(op, Onnx::CustomOperators::SgdVarUpdate);
}

void SGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // Weight update (matching pytorch implementation):
  //   w <- w - (w * wd + delta) * lr

  // First update weights with weight decay
  popops::scaledSubtractFrom(
      graph(),
      get(inId(SGDVarUpdateOp::getVarInIndex())), // weights
      get(inId(SGDVarUpdateOp::getVarInIndex())), // weights
      // wd tensor has already been scaled by lr on the host
      get(inId(SGDVarUpdateOp::getWeightDecayInIndex())),
      prog,
      idStr());

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      get(inId(SGDVarUpdateOp::getVarInIndex())),     // weights
      get(inId(SGDVarUpdateOp::getVarGradInIndex())), // weightDeltas
      get(inId(SGDVarUpdateOp::getLearnRateInIndex())),
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
  //   w <- w - (w * wd + delta) * lr
  //  Or, equivalently:
  //   w <- w * (1 - lr * wd) - lr * delta

  // First update weights with weight decay (only if user has
  // specified non-zero weight decay)
  if (vu_op.getWeightDecay() != 0) {
    popops::scaledSubtractFrom(graph(),
                               get(inId(vu_op.getVarInIndex())), // weights
                               get(inId(vu_op.getVarInIndex())), // weights
                               vu_op.getWeightDecay() * vu_op.getLearnRate(),
                               prog,
                               idStr());

    // TODO: Broadcasting bug in popops prevents the following implementation.
    // when T7138 is complete, investigate whether it is more efficient

    // float weightDecayScaleFactor = 1 - (vu_op.getWeightDecay() *
    // vu_op.getLearnRate());

    // popops::mapInPlace(
    //     graph(),
    //     pe::Mul(pe::_1, pe::Const(weightDecayScaleFactor)),
    //     {get(inId(SGDVarUpdateOp::getVarInIndex()))},
    //     prog,
    //     idStr());
  }

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      get(inId(vu_op.getVarInIndex())),     // weights
      get(inId(vu_op.getVarGradInIndex())), // weightDeltas
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
  poplar::program::Copy copy(get(inId(CopyVarUpdateOp::getVarFromInIndex())),
                             get(inId(CopyVarUpdateOp::getVarToInIndex())));
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
