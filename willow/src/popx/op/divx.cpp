#include <poponnx/error.hpp>
#include <poponnx/op/div.hpp>
#include <poponnx/popx/op/divx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

DivOpx::DivOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<DivOp>(op, {Onnx::Operators::Div_6, Onnx::Operators::Div_7});
}

void DivOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0),
         popops::map(graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     get(inId(DivOp::getArg0InIndex())),
                     get(inId(DivOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

InputCreatorType DivOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) == op_p->outInfo(DivOp::getOutIndex())) {
    return InputCreatorType::CANUNWIND;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor
DivOpx::unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
  return tensor;
}

namespace {
OpxCreator<DivOpx> divOpxCreator({Onnx::Operators::Div_6,
                                  Onnx::Operators::Div_7});
OpxCreator<Opx> divArg0OpxCreator(
    Onnx::GradOperators::DivArg0Grad,
    "DivArg0Grad should be optimised out, \"DivArg0Grad\" pattern is required");
OpxCreator<Opx> divArg1OpxCreator(Onnx::GradOperators::DivArg1Grad,
                                  "DivArg1Grad should be optimised out, "
                                  "\"DivArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace poponnx
