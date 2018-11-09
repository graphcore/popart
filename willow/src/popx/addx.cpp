#include <poponnx/add.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/addx.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popops/ElementWise.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

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

AddGradOpx::AddGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op_p->opType != OpType::ADDGRAD) {
    throw error("cannot create AddGradOpx from " + op_p->op_type());
  }
}

AddGradOp *AddGradOpx::getAddGradOp() const {
  return dynamic_cast<AddGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
