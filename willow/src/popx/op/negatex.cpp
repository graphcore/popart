// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/negatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class NegateGradOp;
class NegateOp;
class Op;

namespace popx {
class Devicex;

NegateOpx::NegateOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateOp>(op, Onnx::Operators::Neg_6);
}

void NegateOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(0,
               snap::Tensor{popops::map(graph().getPoplarGraph(),
                                        popops::expr::UnaryOpType::NEGATE,
                                        getInTensor(0).getPoplarTensor(),
                                        prog.getPoplarSequence(),
                                        debugContext()),
                            graph()});
}

NegateGradOpx::NegateGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NegateGradOp>(op, Onnx::GradOperators::NegGrad);
}

void NegateGradOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(0,
               snap::Tensor{popops::map(graph().getPoplarGraph(),
                                        popops::expr::UnaryOpType::NEGATE,
                                        getInTensor(0).getPoplarTensor(),
                                        prog.getPoplarSequence(),
                                        debugContext()),
                            graph()});
}

namespace {
static OpxCreator<NegateOpx> negOpxCreator(Onnx::Operators::Neg_6);
static OpxCreator<NegateGradOpx>
    negGradOpxCreator(Onnx::GradOperators::NegGrad);
} // namespace

} // namespace popx
} // namespace popart
