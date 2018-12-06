#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/popx/op/sqrtx.hpp>

namespace poponnx {
namespace popx {

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (dynamic_cast<SqrtOp *>(op) == nullptr) {
    throw error("cannot create SqrtOpx from " + op->op_type());
  }
}

void SqrtOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::UnaryOpType::SQRT,
                     get(inId(0)),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
