#include <popops/ElementWise.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/transpose.hpp>
#include <poponnx/popx/op/transposex.hpp>

namespace poponnx {
namespace popx {

TransposeOpx::TransposeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<TransposeOp>()) {
    throw error("cannot create TransposeOpx from " + op->op_type());
  }
}

void TransposeOpx::grow(poplar::program::Sequence &prog) const {
  auto perm = dynamic_cast<TransposeOp *>(op_p)->getPerm();
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
  if (!op->isConvertibleTo<TransposeGradOp>()) {
    throw error("cannot create TransposeGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace poponnx
