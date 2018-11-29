#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/square.hpp>
#include <poponnx/popx/op/squarex.hpp>

namespace poponnx {
namespace popx {

SquareOpx::SquareOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (dynamic_cast<SquareOp *>(op) == nullptr) {
    throw error("cannot create SquareOpx from " + op->op_type());
  }
}

void SquareOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::UnaryOpType::SQUARE,
                     get(inId(0)),
                     prog,
                     idStr()));
}

} // namespace popx
} // namespace poponnx
