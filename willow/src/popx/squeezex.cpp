#include <willow/error.hpp>
#include <willow/popx/squeezex.hpp>
#include <willow/squeeze.hpp>

namespace willow {
namespace popx {

SqueezeOpx::SqueezeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SQUEEZE) {
    throw error("cannot create SqueezeOpx from " + op->op_type());
  }
}

SqueezeOp *SqueezeOpx::getSqueezeOp() const {
  return dynamic_cast<SqueezeOp *>(getOp());
}

SqueezeGradOpx::SqueezeGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SQUEEZEGRAD) {
    throw error("cannot create SqueezeGradOpx from " + op->op_type());
  }
}

SqueezeGradOp *SqueezeGradOpx::getSqueezeGradOp() const {
  return dynamic_cast<SqueezeGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
