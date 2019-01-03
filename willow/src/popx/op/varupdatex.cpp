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
  auto vu_op = getOp<SGDVarUpdateOp>();
  // TODO: When T6033 is complete, change to a single popops::scaledSubtractFrom
  // call for a more efficient implementation

  // Weight deltas, scaled by learning rate, are subtracted from weights.
  // This is implemented as a scaled add with a negative learning rate.
  auto negLr = popops::neg(
      graph(), get(inId(vu_op.getLearnRateInIndex())), prog, idStr());

  popops::scaledAddTo(graph(),
                      get(inId(vu_op.getVarInIndex())),     // weights
                      get(inId(vu_op.getVarGradInIndex())), // weightDeltas
                      negLr,
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
  popops::scaledAddTo(graph(),
                      get(inId(vu_op.getVarInIndex())),     // weights
                      get(inId(vu_op.getVarGradInIndex())), // weightDeltas
                      -1.0f * (vu_op.getLearnRate()),
                      prog,
                      idStr());

  // no poplar::Tensors to insert
}

namespace {
OpxCreator<SGDVarUpdateOpx>
    sgdVarUpdateOpxCreator(Onnx::CustomOperators::SgdVarUpdate);
OpxCreator<ConstSGDVarUpdateOpx>
    constSgdVarUpdateOpxCreator(Onnx::CustomOperators::ConstSgdVarUpdate);
} // namespace

} // namespace popx
} // namespace poponnx
