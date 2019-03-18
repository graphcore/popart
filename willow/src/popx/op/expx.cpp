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

ExpInplaceOpx::ExpInplaceOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ExpInplaceOp>(op, Onnx::CustomOperators::ExpInplace);
}

InputCreatorType ExpInplaceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor ExpInplaceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                 InIndex,
                                                 OutIndex) const {
  return tensor;
}

void ExpInplaceOpx::grow(poplar::program::Sequence &prog) const {

  auto t0 = getInTensor(0);

  // if all of the elements in the tensor are distinct in memory,
  // them we can use the poplar inplace version. Otherwise, we must
  // use a non-inplace version.  See T7110 for a possible improvement
  if (t0.isParallelWriteable()) {
    popops::mapInPlace(graph(),
                       popops::expr::UnaryOpType::EXPONENT,
                       getInTensor(ExpOp::getInIndex()),
                       prog,
                       idStr());

    setOutTensor(0, getInTensor(0));
  }

  else {
    setOutTensor(ExpOp::getOutIndex(),
                 popops::map(graph(),
                             popops::expr::UnaryOpType::EXPONENT,
                             getInTensor(ExpOp::getInIndex()),
                             prog,
                             idStr()));
  }
}

void ExpOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(ExpOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::EXPONENT,
                           getInTensor(ExpOp::getInIndex()),
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
OpxCreator<ExpInplaceOpx>
    expxInplaceOpxCreator(Onnx::CustomOperators::ExpInplace);
OpxCreator<Opx>
    expGradOpxCreator(Onnx::GradOperators::ExpGrad,
                      "ExpGradOp should be removed by pattern 'ExpGradOp'");
} // namespace

} // namespace popx
} // namespace poponnx
