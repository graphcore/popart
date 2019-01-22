#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/scalex.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

ScaleOpx::ScaleOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ScaleOp>(op);
}

void ScaleOpx::grow(poplar::program::Sequence &prog) const {
  auto scale_op     = getOp<ScaleOp>();
  auto scale_factor = static_cast<double>(scale_op.getScaleFactor());
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
  verifyOp<ScaleGradOp>(op, Onnx::GradOperators::ScaleGrad);
}

namespace {
OpxCreator<ScaleOpx> scaleOpxCreator(Onnx::Operators::Scale);
OpxCreator<ScaleGradOpx> scaleGradOpxCreator(Onnx::GradOperators::ScaleGrad);
} // namespace

} // namespace popx
} // namespace poponnx
