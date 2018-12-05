#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/popx/op/sinx.hpp>

namespace poponnx {
namespace popx {

SinOpx::SinOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<SinOp>()) {
    throw error("cannot create SinOpx from " + op->op_type());
  }
}

void SinOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(SinOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::SIN,
                     get(inId(SinOp::getInIndex())),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
