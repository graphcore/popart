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

InputCreatorType ExpOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) == op_p->outInfo(ExpOp::getOutIndex())) {
    return InputCreatorType::CANUNWIND;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor
ExpOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

namespace {
OpxCreator<ExpOpx> expOpxCreator(Onnx::Operators::Exp_6);
OpxCreator<Opx>
    expGradOpxCreator(Onnx::GradOperators::ExpGrad,
                      "ExpGradOp should be removed by pattern 'ExpGradOp'");
} // namespace

} // namespace popx
} // namespace poponnx
