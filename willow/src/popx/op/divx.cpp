#include <poponnx/error.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/popx/op/divx.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

DivOpx::DivOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<DivOp>()) {
    throw error("cannot create DivOpx from " + op->op_type());
  }
}

void DivOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     get(inId(DivOp::getArg0InIndex())),
                     get(inId(DivOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

DivOp *DivOpx::getDivOp() const { return dynamic_cast<DivOp *>(op_p); }

} // namespace popx
} // namespace poponnx
