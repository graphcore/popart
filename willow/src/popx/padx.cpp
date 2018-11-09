#include <poponnx/error.hpp>
#include <poponnx/pad.hpp>
#include <poponnx/popx/padx.hpp>

namespace willow {
namespace popx {

PadOpx::PadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::PAD) {
    throw error("cannot create PadOpx from " + op->op_type());
  }
}

PadOp *PadOpx::getPadOp() const { return dynamic_cast<PadOp *>(op_p); }

} // namespace popx
} // namespace willow
