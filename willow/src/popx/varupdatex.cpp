#include <poponnx/error.hpp>
#include <poponnx/popx/varupdatex.hpp>
#include <poponnx/varupdate.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popops/ScaledAdd.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
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

void ConstSGDVarUpdateOpx::grow() const {
  auto vu_op = getConstSGDVarUpdateOp();
  popops::scaledAddTo(graph(),
                      get(inId(vu_op->getVarIndex())),     // weights
                      get(inId(vu_op->getVarGradIndex())), // weightDeltas
                      -1.0f * (vu_op->getLearnRate()),
                      step(),
                      idStr());

  // no poplar::Tensors to insert!
}

} // namespace popx
} // namespace willow
