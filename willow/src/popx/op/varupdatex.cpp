#include <poponnx/error.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/popx/op/varupdatex.hpp>

#include <popops/ScaledAdd.hpp>

namespace poponnx {
namespace popx {

SGDVarUpdateOpx::SGDVarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SGDVARUPDATE) {
    throw error("cannot create SGDVarUpdateOpx from " + op->op_type());
  }
}

SGDVarUpdateOp *SGDVarUpdateOpx::getSGDVarUpdateOp() const {
  return dynamic_cast<SGDVarUpdateOp *>(op_p);
}

ConstSGDVarUpdateOpx::ConstSGDVarUpdateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::CONSTSGDVARUPDATE) {
    throw error("cannot create ConstSGDVarUpdateOpx from " + op->op_type());
  }
}

ConstSGDVarUpdateOp *ConstSGDVarUpdateOpx::getConstSGDVarUpdateOp() const {
  return dynamic_cast<ConstSGDVarUpdateOp *>(op_p);
}

void ConstSGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getConstSGDVarUpdateOp();
  popops::scaledAddTo(graph(),
                      get(inId(vu_op->getVarInIndex())),     // weights
                      get(inId(vu_op->getVarGradInIndex())), // weightDeltas
                      -1.0f * (vu_op->getLearnRate()),
                      prog,
                      idStr());

  // no poplar::Tensors to insert!
}

} // namespace popx
} // namespace poponnx
