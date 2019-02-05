#include <poponnx/error.hpp>
#include <poponnx/op/mul.hpp>
#include <poponnx/popx/op/mulx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

MulOpx::MulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<MulOp>(op, {Onnx::Operators::Mul_6, Onnx::Operators::Mul_7});
}

void MulOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     get(inId(0)),
                     get(inId(1)),
                     prog,
                     idStr()));
}

InputCreatorType MulOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) == op_p->outInfo(MulOp::getOutIndex())) {
    return InputCreatorType::CANUNWIND;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor
MulOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

namespace {
static OpxCreator<MulOpx> mulOpxCreator({Onnx::Operators::Mul_6,
                                         Onnx::Operators::Mul_7});

static OpxCreator<Opx>
    mulArg0GradOpxCreator(Onnx::GradOperators::MulArg0Grad,
                          "MulArg0GradOp should be optimised out, "
                          "\"MulArgGradOp\" pattern is required");
static OpxCreator<Opx>
    mulArg1GradOpxCreator(Onnx::GradOperators::MulArg1Grad,
                          "MulArg1GradOp should be optimised out, "
                          "\"MulArgGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
