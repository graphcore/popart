#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/sin.hpp>
#include <poponnx/popx/op/sinx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

SinOpx::SinOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SinOp>(op, Onnx::Operators::Sin_7);
}

void SinOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(SinOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::SIN,
                           getInTensor(SinOp::getInIndex()),
                           prog,
                           idStr()));
}

namespace {
OpxCreator<SinOpx> sinOpxCreator(Onnx::Operators::Sin_7);
OpxCreator<Opx> sinGradOpxCreator(
    Onnx::GradOperators::SinGrad,
    "SinGradOp should be optimised out, \"SinGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
