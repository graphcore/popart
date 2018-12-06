#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/tanh.hpp>
#include <poponnx/popx/op/tanhx.hpp>

namespace poponnx {
namespace popx {

TanhOpx::TanhOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<TanhOp>()) {
    throw error("cannot create TanhOpx from " + op->op_type());
  }
}

void TanhOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(TanhOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::TANH,
                     get(inId(TanhOp::getInIndex())),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
