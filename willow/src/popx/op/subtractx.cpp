#include <poponnx/error.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/popx/op/subtractx.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SUBTRACT) {
    throw error("cannot create SubtractOpx from " + op->op_type());
  }
}

void SubtractOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(SubtractOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::SUBTRACT,
                     get(inId(SubtractOp::getArg0InIndex())),
                     get(inId(SubtractOp::getArg1InIndex())),
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

} // namespace popx
} // namespace poponnx
