// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/abs.hpp>
#include <popart/popx/op/absx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

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
