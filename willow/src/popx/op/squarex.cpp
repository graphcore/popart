// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/squarex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;
class SquareOp;

namespace popx {
class Devicex;

SquareOpx::SquareOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SquareOp>(op, Onnx::CustomOperators::Square);
}

void SquareOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::SQUARE,
                           getInTensor(0),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<SquareOpx> squareOpxCreator(Onnx::CustomOperators::Square);
// OpxCreator<Opx> squareGradOpxCreator("SquareGrad", "SquareGradOp should be
// removed by pattern 'SqrtGradOp'");

} // namespace

} // namespace popx
} // namespace popart
