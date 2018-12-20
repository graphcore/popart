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

TransposeGradOpx::TransposeGradOpx(Op *op, Devicex *devicex)
    : TransposeOpx(op, devicex) {
  verifyOp<TransposeGradOp>(op, Onnx::GradOperators::TransposeGrad);
}

namespace {
OpxCreator<TransposeOpx> transposeOpxCreator(Onnx::Operators::Transpose);
OpxCreator<TransposeGradOpx>
    transposeGradOpxCreator(Onnx::GradOperators::TransposeGrad);
} // namespace

} // namespace popx
} // namespace poponnx
