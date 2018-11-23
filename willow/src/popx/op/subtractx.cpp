#include <poponnx/error.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/popx/op/subtractx.hpp>

#include <popops/ElementWise.hpp>

namespace willow {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SUBTRACT) {
    throw error("cannot create SubtractOpx from " + op->op_type());
  }
}

void SubtractOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::SUBTRACT,
                     get(inId(0)),
                     get(inId(1)),
                     prog,
                     idStr()));
}

SubtractOp *SubtractOpx::getSubtractOp() const {
  return dynamic_cast<SubtractOp *>(op_p);
}

SubtractArg0GradOpx::SubtractArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  if (op_p->opType != OpType::SUBTRACTARG0GRAD) {
    throw error("cannot create SubtractArg0GradOpx from " + op_p->op_type());
  }
}

SubtractArg1GradOpx::SubtractArg1GradOpx(Op *op, Devicex *devicex)
    : NegateOpx(op, devicex) {
  if (op_p->opType != OpType::SUBTRACTARG1GRAD) {
    throw error("cannot create SubtractArg1GradOpx from " + op_p->op_type());
  }
}

} // namespace popx
} // namespace willow
