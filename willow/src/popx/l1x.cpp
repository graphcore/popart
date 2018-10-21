#include <willow/error.hpp>
#include <willow/l1.hpp>
#include <willow/popx/l1x.hpp>

namespace willow {
namespace popx {

L1Opx::L1Opx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::L1) {
    throw error("cannot create L1Opx from " + op->op_type());
  }
}

L1Op *L1Opx::getL1Op() const { return dynamic_cast<L1Op *>(getOp()); }

L1GradOpx::L1GradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::L1GRAD) {
    throw error("cannot create L1GradOpx from " + op->op_type());
  }
}

L1GradOp *L1GradOpx::getL1GradOp() const {
  return dynamic_cast<L1GradOp *>(getOp());
}

} // namespace popx
} // namespace willow
