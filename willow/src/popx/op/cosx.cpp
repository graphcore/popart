#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/cos.hpp>
#include <popart/popx/op/cosx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

CosOpx::CosOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<CosOp>(op, Onnx::Operators::Cos_7);
}

void CosOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(CosOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::COS,
                           getInTensor(CosOp::getInIndex()),
                           prog,
                           debugPrefix()));
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
} // namespace popart
