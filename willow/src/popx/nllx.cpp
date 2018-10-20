#include <willow/error.hpp>
#include <willow/nll.hpp>
#include <willow/popx/nllx.hpp>

namespace willow {
namespace popx {

NllOpx::NllOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::NLL) {
    throw error("cannot create NllOpx from " + op->op_type());
  }
}

NllOp *NllOpx::getNllOp() const { return dynamic_cast<NllOp *>(getOp()); }

NllGradOpx::NllGradOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::NLLGRAD) {
    throw error("cannot create NllGradOpx from " + op->op_type());
  }
}

NllGradOp *NllGradOpx::getNllGradOp() const {
  return dynamic_cast<NllGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
