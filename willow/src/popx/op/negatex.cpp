// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/negate.hpp>
#include <popart/popx/op/negatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

NegateOpx::NegateOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateOp>(op, Onnx::Operators::Neg_6);
}

void NegateOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph().getPoplarGraph(),
                           popops::expr::UnaryOpType::NEGATE,
                           getInTensor(0),
                           prog,
                           debugContext()));
}

NegateGradOpx::NegateGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateGradOp>(op, Onnx::GradOperators::NegGrad);
}

void NegateGradOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph().getPoplarGraph(),
                           popops::expr::UnaryOpType::NEGATE,
                           getInTensor(0),
                           prog,
                           debugContext()));
}

namespace {
static OpxCreator<NegateOpx> negOpxCreator(Onnx::Operators::Neg_6);
static OpxCreator<NegateGradOpx>
    negGradOpxCreator(Onnx::GradOperators::NegGrad);
} // namespace

} // namespace popx
} // namespace popart
