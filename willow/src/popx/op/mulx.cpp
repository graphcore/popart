#include <poponnx/error.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/popx/op/mulx.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

MulOpx::MulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<MulOp>()) {
    throw error("cannot create MulOpx from " + op->op_type());
  }
}

void MulOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     get(inId(0)),
                     get(inId(1)),
                     prog,
                     idStr()));
}

MulOp *MulOpx::getMulOp() const { return dynamic_cast<MulOp *>(op_p); }

} // namespace popx
} // namespace poponnx
