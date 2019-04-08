#include <poponnx/op/elementwise.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {
namespace popx {

ElementWiseUnaryOutplaceOpx::ElementWiseUnaryOutplaceOpx(
    Op *op,
    Devicex *devx,
    std::unique_ptr<EwuComputex> cx_)
    : ElementWiseUnaryOpx(op, devx), cx(std::move(cx_)) {}

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

void ElementWiseUnaryInplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = getInTensor(ElementWiseUnaryOp::getInIndex());

  // if all of the elements in the tensor are distinct in memory,
  // them we can use the poplar inplace version. Otherwise, we must
  // use a non-inplace version.  See T7110 for a possible improvement
  if (!outTensor.isParallelWriteable()) {
    outTensor = cx->outplace(prog, graph(), outTensor);
  } else {
    cx->inplace(prog, graph(), outTensor);
  }
  setOutTensor(ElementWiseUnaryOp::getOutIndex(), outTensor);
}

void ElementWiseUnaryOutplaceOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cx->outplace(
      prog, graph(), getInTensor(ElementWiseUnaryOp::getInIndex()));

  setOutTensor(ElementWiseUnaryOp::getOutIndex(), outTensor);
}

poplar::Tensor ElementWiseBinaryOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                        InIndex,
                                                        OutIndex) const {
  return tensor;
}

poplar::Tensor EwuComputex::cloneNcopy(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor) const {

  auto outTensor = graph.clone(tensor);
  poplar::program::Copy copyProg(tensor, outTensor);
  prog.add(copyProg);
  return outTensor;
}

} // namespace popx
} // namespace poponnx
