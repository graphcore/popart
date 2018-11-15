#include <poponnx/add.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/addx.hpp>

#include <popops/ElementWise.hpp>

namespace willow {
namespace popx {

AddOpx::AddOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::ADD) {
    throw error("cannot create AddOpx from " + op->op_type());
  }
}

void AddOpx::grow() const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::ADD,
                     get(inId(0)),
                     get(inId(1)),
                     step(),
                     idStr()));
}

AddOp *AddOpx::getAddOp() const { return dynamic_cast<AddOp *>(op_p); }

AddArg0GradOpx::AddArg0GradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (op_p->opType != OpType::ADDARG0GRAD) {
    throw error("cannot create AddArg0GradOpx from " + op_p->op_type());
  }
}

AddArg1GradOpx::AddArg1GradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (op_p->opType != OpType::ADDARG1GRAD) {
    throw error("cannot create AddArg1GradOpx from " + op_p->op_type());
  }
}

} // namespace popx
} // namespace willow
