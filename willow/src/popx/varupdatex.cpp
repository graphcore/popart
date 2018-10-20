#include <willow/error.hpp>
#include <willow/popx/varupdatex.hpp>
#include <willow/varupdate.hpp>

namespace willow {
namespace popx {

VarUpdateOpx::VarUpdateOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::VARUPDATE) {
    throw error("cannot create VarUpdateOpx from " + op->op_type());
  }
}

VarUpdateOp *VarUpdateOpx::getVarUpdateOp() const {
  return dynamic_cast<VarUpdateOp *>(getOp());
}

} // namespace popx
} // namespace willow
