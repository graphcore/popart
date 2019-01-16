#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/popx/op/expx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

ExpOpx::ExpOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ExpOp>(op, Onnx::Operators::Exp_6);
}

void ExpOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(ExpOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::EXPONENT,
                     get(inId(ExpOp::getInIndex())),
                     prog,
                     idStr()));
}

namespace {
OpxCreator<ExpOpx> expOpxCreator(Onnx::Operators::Exp_6);
OpxCreator<Opx>
    expGradOpxCreator(Onnx::GradOperators::ExpGrad,
                      "ExpGradOp should be removed by pattern 'ExpGradOp'");
} // namespace

} // namespace popx
} // namespace poponnx
