#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/transpose.hpp>
#include <poponnx/popx/op/transposex.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

TransposeOpx::TransposeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TransposeOp>(op);
}

void TransposeOpx::grow(poplar::program::Sequence &prog) const {
  auto perm = getOp<TransposeOp>().getPerm();
  std::vector<unsigned> unsigned_perm;
  for (auto i : perm) {
    unsigned_perm.push_back(static_cast<unsigned>(i));
  }

  auto input      = get(inId(TransposeOp::getInIndex()));
  auto input_copy = cloneNcopy(prog, input);
  auto output     = input_copy.dimShuffle(unsigned_perm);
  insert(outId(TransposeOp::getOutIndex()), output);
}

InputCreatorType TransposeOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CANUNWIND;
}

poplar::Tensor TransposeOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                InIndex,
                                                OutIndex) const {
  auto perm = getOp<TransposeOp>().getPerm();
  std::vector<unsigned> reverse_perm;

  // For each dimension, find its position in perm
  for (int i = 0; i < perm.size(); i++) {
    auto it       = std::find(perm.begin(), perm.end(), i);
    auto position = std::distance(perm.begin(), it);
    reverse_perm.push_back(static_cast<unsigned>(position));
  }

  return tensor.dimShuffle(reverse_perm);
}

TransposeGradOpx::TransposeGradOpx(Op *op, Devicex *devicex)
    : TransposeOpx(op, devicex) {
  verifyOp<TransposeGradOp>(op, Onnx::GradOperators::TransposeGrad);
}

namespace {
OpxCreator<TransposeOpx> transposeOpxCreator(Onnx::Operators::Transpose_1);
OpxCreator<TransposeGradOpx>
    transposeGradOpxCreator(Onnx::GradOperators::TransposeGrad);
} // namespace

} // namespace popx
} // namespace poponnx
