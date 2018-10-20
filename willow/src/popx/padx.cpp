#include <willow/error.hpp>
#include <willow/pad.hpp>
#include <willow/popx/padx.hpp>

namespace willow {
namespace popx {

PadOpx::PadOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::PAD) {
    throw error("cannot create PadOpx from " + op->op_type());
  }
}

PadOp *PadOpx::getPadOp() const { return dynamic_cast<PadOp *>(getOp()); }

} // namespace popx
} // namespace willow
