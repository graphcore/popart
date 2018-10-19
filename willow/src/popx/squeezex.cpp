#include <willow/error.hpp>
#include <willow/popx/squeezex.hpp>
#include <willow/squeeze.hpp>

namespace willow {
namespace popx {

SqueezeOpx::SqueezeOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::SQUEEZE) {
    throw error("cannot create SqueezeOpx from " + op->op_type());
  }
}

SqueezeOp *SqueezeOpx::getSqueezeOp() const {
  return dynamic_cast<SqueezeOp *>(getOp());
}

} // namespace popx
} // namespace willow
