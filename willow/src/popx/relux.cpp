#include <willow/error.hpp>
#include <willow/popx/relux.hpp>
#include <willow/relu.hpp>

namespace willow {
namespace popx {

ReluOpx::ReluOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELU) {
    throw error("cannot create ReluOpx from " + op->op_type());
  }
}

ReluOp *ReluOpx::getReluOp() const { return dynamic_cast<ReluOp *>(getOp()); }

ReluGradOpx::ReluGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::RELUGRAD) {
    throw error("cannot create ReluGradOpx from " + op->op_type());
  }
}

ReluGradOp *ReluGradOpx::getReluGradOp() const {
  return dynamic_cast<ReluGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
