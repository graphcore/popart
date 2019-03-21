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

  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::BinaryOpType::MULTIPLY,
                           scale_factor_const,
                           getInTensor(0),
                           prog,
                           idStr()));
}

ScaleGradOpx::ScaleGradOpx(Op *op, Devicex *devicex) : ScaleOpx(op, devicex) {
  verifyOp<ScaleGradOp>(op, Onnx::GradOperators::ScaleGrad);
}

void ScaleInplaceOpx::grow(poplar::program::Sequence &prog) const {

  auto scale_inplace_op = getOp<ScaleInplaceOp>();
  auto scale_factor = static_cast<double>(scale_inplace_op.getScaleFactor());
  auto scale_factor_const =
      dv_p->getConst(popType(op_p->inInfo(0)), {1}, scale_factor);

  auto t0 = getInTensor(0);

  // if all of the elements in the tensor are distinct in memory,
  // them we can use the poplar inplace version. Otherwise, we must
  // use a non-inplace version.  See T7110 for a possible improvement
  if (t0.isParallelWriteable()) {
    popops::mapInPlace(graph(),
                       popops::expr::BinaryOpType::MULTIPLY,
                       getInTensor(0),
                       scale_factor_const,
                       prog,
                       idStr());
    setOutTensor(0, getInTensor(0));
  }

  else {
    setOutTensor(0,
                 popops::map(graph(),
                             popops::expr::BinaryOpType::MULTIPLY,
                             scale_factor_const,
                             getInTensor(0),
                             prog,
                             idStr()));
  }
}

InputCreatorType ScaleInplaceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor ScaleInplaceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                   InIndex,
                                                   OutIndex) const {
  return tensor;
}

ScaleInplaceOpx::ScaleInplaceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ScaleInplaceOp>(op, Onnx::CustomOperators::ScaleInplace);
}

namespace {
OpxCreator<ScaleOpx> scaleOpxCreator(Onnx::Operators::Scale_1);
OpxCreator<ScaleInplaceOpx>
    scalexInplaceOpxCreator(Onnx::CustomOperators::ScaleInplace);
OpxCreator<ScaleGradOpx> scaleGradOpxCreator(Onnx::GradOperators::ScaleGrad);
} // namespace

} // namespace popx
} // namespace poponnx
