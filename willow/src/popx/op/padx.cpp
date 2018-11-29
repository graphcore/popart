#include <poponnx/error.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/popx/op/padx.hpp>

namespace poponnx {
namespace popx {

PadOpx::PadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::PAD) {
    throw error("cannot create PadOpx from " + op->op_type());
  }
}

PadOp *PadOpx::getPadOp() const { return dynamic_cast<PadOp *>(op_p); }

} // namespace popx
} // namespace poponnx
