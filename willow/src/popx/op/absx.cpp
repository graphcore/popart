// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/abs.hpp>
#include <popart/popx/op/absx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

AbsOpx::AbsOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<AbsOp>(op, {Onnx::Operators::Abs_6});
}

void AbsOpx::grow(snap::program::Sequence &prog) const {

  setOutTensor(
      AbsOp::getOutIndex(),
      snap::Tensor{
          popops::map(graph().getPoplarGraph(),
                      popops::expr::UnaryOpType::ABSOLUTE,
                      getInTensor(AbsOp::getInIndex()).getPoplarTensor(),
                      prog.getPoplarSequence(),
                      debugContext()),
          graph()});
}

namespace {
OpxCreator<AbsOpx> absOpxCreator(Onnx::Operators::Abs_6);
} // namespace

} // namespace popx
} // namespace popart
