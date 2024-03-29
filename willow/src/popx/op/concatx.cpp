// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/concat.hpp>
#include <popart/popx/op/concatx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
class Devicex;

BaseConcatOpx::BaseConcatOpx(Op *op_, Devicex *devicex)
    : Opx(op_, devicex), op(static_cast<ConcatOp *>(op_)) {}

ConcatOpx::ConcatOpx(Op *op_, Devicex *devicex)
    : BaseConcatOpx(op_, devicex), op(static_cast<ConcatOp *>(op_)) {
  verifyOp<ConcatOp>(op_,
                     {Onnx::Operators::Concat_1,
                      Onnx::Operators::Concat_4,
                      Onnx::Operators::Concat_11});
}

InputCreatorType BaseConcatOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor BaseConcatOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                 InIndex inIndex,
                                                 OutIndex) const {
  int64_t start = 0L;
  for (int i = 0; i < inIndex; ++i) {
    auto shape = op->inShape(ConcatOp::getInIndex(i));
    start += shape[op->getAxis()];
  }
  int64_t end = 0L;
  for (int i = 0; i <= inIndex; ++i) {
    auto shape = op->inShape(ConcatOp::getInIndex(i));
    end += shape[op->getAxis()];
  }
  return tensor.slice(static_cast<std::size_t>(start),
                      static_cast<std::size_t>(end),
                      static_cast<unsigned>(op->getAxis()));
}

view::RegMap BaseConcatOpx::unwindRegion(InIndex inIndex,
                                         OutIndex outIndex) const {
  ConcatOp *cop = dynamic_cast<ConcatOp *>(this->op_p);
  return cop->bwdRegMap(inIndex, outIndex);
}

void ConcatOpx::grow(poplar::program::Sequence &prog) const {
  std::vector<poplar::Tensor> tensors;
  tensors.reserve(op->input->n());

  for (int i = 0; i < op->input->n(); ++i) {
    tensors.push_back(getInTensor(ConcatOp::getInIndex(i)));
  }

  poplar::Tensor concat =
      poplar::concat(tensors, static_cast<unsigned>(op->getAxis()));

  setOutTensor(ConcatOp::getOutIndex(), cloneNcopy(prog, concat));
}

ConcatInplaceOpx::ConcatInplaceOpx(Op *op_, Devicex *devicex)
    : BaseConcatOpx(op_, devicex), op(static_cast<ConcatOp *>(op_)) {
  verifyOp<ConcatOp>(op_);
}

void ConcatInplaceOpx::grow(poplar::program::Sequence &) const {
  std::vector<poplar::Tensor> tensors;
  tensors.reserve(op->input->n());

  for (int i = 0; i < op->input->n(); ++i) {
    tensors.push_back(getInTensor(ConcatOp::getInIndex(i)));
  }

  poplar::Tensor concat =
      poplar::concat(tensors, static_cast<unsigned>(op->getAxis()));

  setOutTensor(ConcatOp::getOutIndex(), concat);
}

ConcatGradOpx::ConcatGradOpx(Op *op_, Devicex *devicex)
    : Opx(op_, devicex), op(static_cast<ConcatGradOp *>(op_)) {
  verifyOp<ConcatGradOp>(op_, Onnx::GradOperators::ConcatGrad);
}

void ConcatGradOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(ConcatGradOp::getInIndex());
  auto out   = input.slice(static_cast<std::size_t>(op->getStart()),
                         static_cast<std::size_t>(op->getEnd()),
                         static_cast<unsigned>(op->getAxis()));

  setOutTensor(ConcatGradOp::getOutIndex(), cloneNcopy(prog, out));
}

namespace {
OpxCreator<ConcatOpx> concatOpxCreator({Onnx::Operators::Concat_1,
                                        Onnx::Operators::Concat_4,
                                        Onnx::Operators::Concat_11});
OpxCreator<ConcatInplaceOpx>
    concatInplaceOpxCreator(Onnx::CustomOperators::ConcatInplace);
OpxCreator<ConcatGradOpx> concatGradOpxCreator(Onnx::GradOperators::ConcatGrad);
} // namespace

} // namespace popx
} // namespace popart
