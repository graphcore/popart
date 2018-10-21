#include <willow/add.hpp>
#include <willow/error.hpp>
#include <willow/popx/addx.hpp>

namespace willow {
namespace popx {

AddOpx::AddOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::ADD) {
    throw error("cannot create AddOpx from " + op->op_type());
  }
}

AddOp *AddOpx::getAddOp() const { return dynamic_cast<AddOp *>(getOp()); }

AddGradOpx::AddGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::ADDGRAD) {
    throw error("cannot create AddGradOpx from " + op->op_type());
  }
}

AddGradOp *AddGradOpx::getAddGradOp() const {
  return dynamic_cast<AddGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
