#include <popops/Cast.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/cast.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/castx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

CastOpx::CastOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CastOp>(op);
}

void CastOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(CastOp::getOutIndex(),
               popops::cast(graph(),
                            getInTensor(CastOp::getInIndex()),
                            popType(op_p->outInfo(CastOp::getOutIndex())),
                            prog,
                            idStr()));
}

CastGradOpx::CastGradOpx(Op *op, Devicex *devicex) : CastOpx(op, devicex) {
  verifyOp<CastGradOp>(op, Onnx::GradOperators::CastGrad);
}

namespace {
OpxCreator<CastOpx> castOpxCreator(Onnx::Operators::Cast_9);
OpxCreator<CastGradOpx> castGradOpxCreator(Onnx::GradOperators::CastGrad);
} // namespace

} // namespace popx
} // namespace poponnx
