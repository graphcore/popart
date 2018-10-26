#include <willow/add.hpp>
#include <willow/error.hpp>
#include <willow/popx/addx.hpp>
#include <willow/popx/devicex.hpp>


#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popops/Expr.hpp>
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

  dv_p->insert(op_p->output.id(0),
          popops::map(dv_p->graph(),
                      popops::expr::BinaryOpType::ADD,
                      dv_p->getTensor(op_p->input.id(0)),
                      dv_p->getTensor(op_p->input.id(1)),
                      dv_p->progs.step(),
                      std::to_string(op_p->id)));
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
