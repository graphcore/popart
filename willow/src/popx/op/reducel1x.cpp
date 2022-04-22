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
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/reducel1.hpp>
#include <popart/popx/op/reducel1x.hpp>
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

ReduceL1Opx::ReduceL1Opx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ReduceL1Op>(op);
}

void ReduceL1Opx::grow(snap::program::Sequence &prog) const {
  const auto &op   = getOp<ReduceL1Op>();
  const auto input = getInTensor(ReduceL1Op::getInIndex()).getPoplarTensor();

  auto abs_input = popops::abs(graph().getPoplarGraph(),
                               input,
                               prog.getPoplarSequence(),
                               debugContext("abs"));

  auto output_tensor = popops::reduce(graph().getPoplarGraph(),
                                      abs_input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::ADD},
                                      prog.getPoplarSequence(),
                                      debugContext("reduce"));

  setOutTensor(ReduceL1Op::getOutIndex(),
               snap::Tensor{output_tensor.reshape(
                                outInfo(ReduceL1Op::getOutIndex()).shape_szt()),
                            graph()});
}

ReduceL1GradOpx::ReduceL1GradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceL1GradOp>(op, Onnx::GradOperators::ReduceL1Grad);
}

void ReduceL1GradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op       = getOp<ReduceL1GradOp>();
  auto output          = getInTensor(ReduceL1GradOp::getOutIndex());
  auto fwd_input       = getInTensor(ReduceL1GradOp::getFwdInInIndex());
  auto input_shape     = inShape(ReduceL1GradOp::getInIndex());
  auto output_shape    = outShape(ReduceL1GradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output = snap::popops::map(graph(),
                             pe::Mul(pe::_1, pe::Signum(pe::_2)),
                             {output, fwd_input},
                             prog,
                             debugContext("output"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceL1GradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceL1Opx> reduceL1OpxCreator({Onnx::Operators::ReduceL1_1,
                                            Onnx::Operators::ReduceL1_11});
OpxCreator<ReduceL1GradOpx>
    reduceL1GradGradOpxCreator(Onnx::GradOperators::ReduceL1Grad);
} // namespace

} // namespace popx
} // namespace popart
