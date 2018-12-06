#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/popx/op/expx.hpp>

namespace poponnx {
namespace popx {

ExpOpx::ExpOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<ExpOp>()) {
    throw error("cannot create ExpOpx from " + op->op_type());
  }
}

void ExpOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(ExpOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::EXPONENT,
                     get(inId(ExpOp::getInIndex())),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
