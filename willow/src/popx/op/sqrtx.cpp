#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/popx/op/sqrtx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SqrtOp>(op, Onnx::Operators::Sqrt_6);
}

void SqrtOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::SQRT,
                           getInTensor(0),
                           prog,
                           debugPrefix()));
}

namespace {
OpxCreator<SqrtOpx> sqrtOpxCreator(Onnx::Operators::Sqrt_6);
OpxCreator<Opx> softmaxGradOpxCreator(
    Onnx::GradOperators::SqrtGrad,
    "SqrtGradOp should be removed by pattern 'SqrtGradOp'");

} // namespace

} // namespace popx
} // namespace popart
