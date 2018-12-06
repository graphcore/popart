#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/scalex.hpp>

namespace poponnx {
namespace popx {

ScaleOpx::ScaleOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<ScaleOp>()) {
    throw error("cannot create ScaleOpx from " + op->op_type());
  }
}

void ScaleOpx::grow(poplar::program::Sequence &prog) const {
  auto scale_op     = dynamic_cast<ScaleOp *>(op_p);
  auto scale_factor = static_cast<double>(scale_op->getScaleFactor());
  auto scale_factor_const =
      dv_p->getConst(popType(op_p->inInfo(0)), {1}, scale_factor);

  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     scale_factor_const,
                     get(inId(0)),
                     prog,
                     idStr()));
}

ScaleGradOpx::ScaleGradOpx(Op *op, Devicex *devicex) : ScaleOpx(op, devicex) {
  if (!op->isConvertibleTo<ScaleGradOp>()) {
    throw error("cannot create ScaleGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace poponnx
