// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/transpose.hpp>
#include <popart/popx/op/transposex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
namespace popx {
class Devicex;

TransposeOpx::TransposeOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TransposeOp>(op);
}

void TransposeOpx::grow(snap::program::Sequence &prog) const {
  auto perm = getOp<TransposeOp>().getPerm();
  std::vector<unsigned> unsigned_perm;
  for (auto i : perm) {
    unsigned_perm.push_back(static_cast<unsigned>(i));
  }

  auto input      = getInTensor(TransposeOp::getInIndex());
  auto input_copy = cloneNcopy(prog, input).getPoplarTensor();
  auto output     = input_copy.dimShuffle(unsigned_perm);
  setOutTensor(TransposeOp::getOutIndex(), snap::Tensor{output, graph()});
}

InputCreatorType TransposeOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor
TransposeOpx::unwindTensorLayout(snap::Tensor tensor, InIndex, OutIndex) const {
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

view::RegMap TransposeOpx::unwindRegion(InIndex inIndex,
                                        OutIndex outIndex) const {
  TransposeOp *op = dynamic_cast<TransposeOp *>(this->op_p);
  return op->bwdRegMap(inIndex, outIndex);
}

TransposeInplaceOpx::TransposeInplaceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<TransposeInplaceOp>(op);
}

InputCreatorType TransposeInplaceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor TransposeInplaceOpx::unwindTensorLayout(snap::Tensor tensor,
                                                     InIndex,
                                                     OutIndex) const {
  auto perm = getOp<TransposeInplaceOp>().getPerm();
  std::vector<unsigned> reverse_perm;

  // For each dimension, find its position in perm
  for (int i = 0; i < perm.size(); i++) {
    auto it       = std::find(perm.begin(), perm.end(), i);
    auto position = std::distance(perm.begin(), it);
    reverse_perm.push_back(static_cast<unsigned>(position));
  }

  return tensor.dimShuffle(reverse_perm);
}

view::RegMap TransposeInplaceOpx::unwindRegion(InIndex inIndex,
                                               OutIndex outIndex) const {
  TransposeInplaceOp *op = dynamic_cast<TransposeInplaceOp *>(this->op_p);
  return op->bwdRegMap(inIndex, outIndex);
}

void TransposeInplaceOpx::grow(snap::program::Sequence &) const {
  auto perm = getOp<TransposeInplaceOp>().getPerm();
  std::vector<unsigned> unsigned_perm;
  for (auto i : perm) {
    unsigned_perm.push_back(static_cast<unsigned>(i));
  }

  setOutTensor(TransposeOp::getOutIndex(),
               snap::Tensor{getInTensor(TransposeOp::getInIndex())
                                .getPoplarTensor()
                                .dimShuffle(unsigned_perm),
                            graph()});
}

TransposeGradOpx::TransposeGradOpx(Op *op, Devicex *devicex)
    : TransposeOpx(op, devicex) {
  verifyOp<TransposeGradOp>(op, Onnx::GradOperators::TransposeGrad);
}

namespace {
OpxCreator<TransposeOpx> transposeOpxCreator(Onnx::Operators::Transpose_1);
OpxCreator<TransposeInplaceOpx>
    transposeInplaceOpxCreator(Onnx::CustomOperators::TransposeInplace);
OpxCreator<TransposeGradOpx>
    transposeGradOpxCreator(Onnx::GradOperators::TransposeGrad);
} // namespace

} // namespace popx
} // namespace popart
