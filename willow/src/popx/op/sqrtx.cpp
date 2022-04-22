// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/sqrtx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Op;
class SqrtOp;

namespace popx {
class Devicex;

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SqrtOp>(op, Onnx::Operators::Sqrt_6);
}

void SqrtOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(0,
               snap::Tensor{popops::map(graph().getPoplarGraph(),
                                        popops::expr::UnaryOpType::SQRT,
                                        getInTensor(0).getPoplarTensor(),
                                        prog.getPoplarSequence(),
                                        debugContext()),
                            graph()});
}

namespace {
OpxCreator<SqrtOpx> sqrtOpxCreator(Onnx::Operators::Sqrt_6);
} // namespace

} // namespace popx
} // namespace popart
