#include <willow/error.hpp>
#include <willow/popx/varupdatex.hpp>
#include <willow/varupdate.hpp>

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

} // namespace popx
} // namespace willow
