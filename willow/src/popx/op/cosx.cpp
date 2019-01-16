#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/cos.hpp>
#include <poponnx/popx/op/cosx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

CosOpx::CosOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CosOp>(op, Onnx::Operators::Cos_7);
}

void CosOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(CosOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::COS,
                     get(inId(CosOp::getInIndex())),
                     prog,
                     idStr()));
}

namespace {
OpxCreator<CosOpx> cosOpxCreator(Onnx::Operators::Cos_7);
OpxCreator<Opx> coshOpxCreator(Onnx::Operators::Cosh_9,
                               "Cosh should be removed by pattern \"CoshOp\"");
OpxCreator<Opx> cosGradOpxCreator(
    Onnx::GradOperators::CosGrad,
    "CosGradOp should be optimised out, \"CosGradOp\" pattern is required");

} // namespace

} // namespace popx
} // namespace poponnx
