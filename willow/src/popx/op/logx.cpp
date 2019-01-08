#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/log.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/logx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

LogOpx::LogOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LogOp>(op, Onnx::Operators::Log);
}

void LogOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = popops::map(graph(),
                               popops::expr::UnaryOpType::LOGARITHM,
                               get(inId(LogOp::getInIndex())),
                               prog,
                               idStr());

  insert(outId(LogOp::getOutIndex()), outTensor);
}

namespace {
OpxCreator<LogOpx> logOpxCreator(Onnx::Operators::Log);
OpxCreator<Opx> logGradOpxCreator(
    Onnx::GradOperators::LogGrad,
    "LogGradOp should be optimised out, \"LogGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
