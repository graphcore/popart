#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sqrt.hpp>
#include <poponnx/popx/op/sqrtx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SqrtOp>(op, Onnx::Operators::Sqrt_6);
}

void SqrtOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::UnaryOpType::SQRT,
                     get(inId(0)),
                     prog,
                     idStr()));
}

namespace {
OpxCreator<SqrtOpx> sqrtOpxCreator(Onnx::Operators::Sqrt_6);
OpxCreator<Opx> softmaxGradOpxCreator(
    Onnx::GradOperators::SqrtGrad,
    "SqrtGradOp should be removed by pattern 'SqrtGradOp'");

} // namespace

} // namespace popx
} // namespace poponnx
