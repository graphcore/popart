#include <poponnx/error.hpp>
#include <poponnx/popx/subtractx.hpp>
#include <poponnx/subtract.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <popops/ElementWise.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SUBTRACT) {
    throw error("cannot create SubtractOpx from " + op->op_type());
  }
}

void SubtractOpx::grow() const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::SUBTRACT,
                     get(inId(0)),
                     get(inId(1)),
                     step(),
                     idStr()));
}

SubtractOp *SubtractOpx::getSubtractOp() const {
  return dynamic_cast<SubtractOp *>(op_p);
}

SubtractGradOpx::SubtractGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op_p->opType != OpType::SUBTRACTGRAD) {
    throw error("cannot create SubtractGradOpx from " + op_p->op_type());
  }
}

SubtractGradOp *SubtractGradOpx::getSubtractGradOp() const {
  return dynamic_cast<SubtractGradOp *>(op_p);
}

} // namespace popx
} // namespace willow
