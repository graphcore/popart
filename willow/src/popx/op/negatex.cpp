#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/popx/op/negatex.hpp>

namespace poponnx {
namespace popx {

NegateOpx::NegateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (dynamic_cast<NegateOp *>(op) == nullptr) {
    throw error("cannot create NegateOpx from " + op->op_type());
  }
}

void NegateOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     get(inId(0)),
                     prog,
                     idStr()));
}

NegateGradOpx::NegateGradOpx(Op *op, Devicex *devicex)
    : NegateOpx(op, devicex) {
  if (dynamic_cast<NegateGradOp *>(op) == nullptr) {
    throw error("cannot create NegateGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace poponnx
