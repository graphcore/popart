#include <poponnx/error.hpp>
#include <poponnx/op/subtract.hpp>
#include <poponnx/popx/op/subtractx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SubtractOp>(op, {Onnx::Operators::Sub_6, Onnx::Operators::Sub_7});
}

void SubtractOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(SubtractOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::SUBTRACT,
                     get(inId(SubtractOp::getArg0InIndex())),
                     get(inId(SubtractOp::getArg1InIndex())),
                     prog,
                     idStr()));
}

SubtractArg0GradOpx::SubtractArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<SubtractArg0GradOp>(op, Onnx::GradOperators::SubArg0Grad);
}

InputCreatorType SubtractOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) == op_p->outInfo(SubtractOp::getOutIndex())) {
    return InputCreatorType::AGNOSTICTOLAYOUT;
  } else {
    return InputCreatorType::DEADEND;
  }
}

namespace {
OpxCreator<SubtractOpx> subtractOpxCreator({Onnx::Operators::Sub_6,
                                            Onnx::Operators::Sub_7});
OpxCreator<SubtractArg0GradOpx>
    subtractArg0GradOpxCreator(Onnx::GradOperators::SubArg0Grad);
OpxCreator<Opx> subtractArg1GradOpxCreator(
    Onnx::GradOperators::SubArg1Grad,
    "SubtractArg1GradOpx should be removed by pattern 'SubtractArg1GradOp'");
} // namespace

} // namespace popx
} // namespace poponnx
