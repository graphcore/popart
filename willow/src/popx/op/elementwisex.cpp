#include <poponnx/op/elementwise.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {
namespace popx {

ElementWiseUnaryOpx::ElementWiseUnaryOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType ElementWiseUnaryOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor ElementWiseUnaryOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                       InIndex,
                                                       OutIndex) const {
  return tensor;
}

ElementWiseBinaryOpx::ElementWiseBinaryOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType
ElementWiseBinaryOpx::getInputCreatorType(InIndex index) const {
  // Check shape doesn't change due to numpy-style broadcasting.
  // Design choice: even without broadcasting, it is possible for the
  // two inputs (of same shape) have different layout.
  // The poplar binary op can choose the layout of the output to take
  // the layout of either input.
  // However, let's layout both inputs in the same way. That way we can
  // definitely unwind through this opx, and it will also be efficient
  // when performing the op.
  if (op_p->inInfo(index) ==
      op_p->outInfo(ElementWiseBinaryOp::getOutIndex())) {
    return InputCreatorType::CANUNWIND;
  } else {
    return InputCreatorType::DEADEND;
  }
}

poplar::Tensor ElementWiseBinaryOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                        InIndex,
                                                        OutIndex) const {
  return tensor;
}

} // namespace popx
} // namespace poponnx
