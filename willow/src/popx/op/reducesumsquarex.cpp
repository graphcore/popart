// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/popx/op/reducesumsquarex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

ReduceSumSquareOpx::ReduceSumSquareOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceSumSquareOp>(op);
}

void ReduceSumSquareOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<ReduceSumSquareOp>();
  const auto input =
      getInTensor(ReduceSumSquareOp::getInIndex()).getPoplarTensor();

  auto output_tensor = popops::reduce(graph().getPoplarGraph(),
                                      input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::SQUARE_ADD},
                                      prog.getPoplarSequence(),
                                      debugContext("squareAdd"));

  setOutTensor(
      ReduceSumSquareOp::getOutIndex(),
      snap::Tensor{output_tensor.reshape(
                       outInfo(ReduceSumSquareOp::getOutIndex()).shape_szt()),
                   graph()});
}

ReduceSumSquareGradOpx::ReduceSumSquareGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceSumSquareGradOp>(op, Onnx::GradOperators::ReduceSumSquareGrad);
}

void ReduceSumSquareGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<ReduceSumSquareGradOp>();
  auto output =
      cloneNcopy(prog, getInTensor(ReduceSumSquareGradOp::getOutIndex()));
  auto input_shape     = inShape(ReduceSumSquareGradOp::getInIndex());
  auto output_shape    = outShape(ReduceSumSquareGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  // TODO: should this be a mapInPlace?
  output = snap::popops::map(
      graph(),
      pe::Mul(pe::Mul(pe::_1, pe::_2), pe::Const(2)),
      {output, getInTensor(ReduceSumSquareGradOp::getFwdInInIndex())},
      prog,
      debugContext("mul"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceSumSquareGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceSumSquareOpx> reduceSumSquareOpxCreator(
    {Onnx::Operators::ReduceSumSquare_1, Onnx::Operators::ReduceSumSquare_11});
OpxCreator<ReduceSumSquareGradOpx>
    reduceSumSquareGradGradOpxCreator(Onnx::GradOperators::ReduceSumSquareGrad);
} // namespace

} // namespace popx
} // namespace popart
