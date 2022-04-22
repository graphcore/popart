// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/reducelogsum.hpp>
#include <popart/popx/op/reducelogsumx.hpp>
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

ReduceLogSumOpx::ReduceLogSumOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceLogSumOp>(op);
}

void ReduceLogSumOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<ReduceLogSumOp>();
  const auto input =
      getInTensor(ReduceLogSumOp::getInIndex()).getPoplarTensor();

  auto output_tensor = popops::reduce(graph().getPoplarGraph(),
                                      input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::ADD},
                                      prog.getPoplarSequence(),
                                      debugContext("output"));
  popops::logInPlace(graph().getPoplarGraph(),
                     output_tensor,
                     prog.getPoplarSequence(),
                     debugContext("log"));

  setOutTensor(
      ReduceLogSumOp::getOutIndex(),
      snap::Tensor{output_tensor.reshape(
                       outInfo(ReduceLogSumOp::getOutIndex()).shape_szt()),
                   graph()});
}

ReduceLogSumGradOpx::ReduceLogSumGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceLogSumGradOp>(op, Onnx::GradOperators::ReduceLogSumGrad);
}

void ReduceLogSumGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<ReduceLogSumGradOp>();
  auto output = getInTensor(ReduceLogSumGradOp::getInIndex()).getPoplarTensor();
  auto scale =
      getInTensor(ReduceLogSumGradOp::getFwdOutInIndex()).getPoplarTensor();
  auto input_shape     = inShape(ReduceLogSumGradOp::getInIndex());
  auto output_shape    = outShape(ReduceLogSumGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);
  scale  = scale.reshape(new_shape);
  scale =
      popops::exp(graph().getPoplarGraph(), scale, prog.getPoplarSequence());

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
      scale  = scale.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output = popops::div(graph().getPoplarGraph(),
                       output,
                       scale,
                       prog.getPoplarSequence(),
                       debugContext("div"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceLogSumGradOp::getOutIndex(),
               snap::Tensor{output, graph()});
}

namespace {
OpxCreator<ReduceLogSumOpx> reduceLogSumOpxCreator(
    {Onnx::Operators::ReduceLogSum_1, Onnx::Operators::ReduceLogSum_11});
OpxCreator<ReduceLogSumGradOpx>
    reduceLogSumGradGradOpxCreator(Onnx::GradOperators::ReduceLogSumGrad);
} // namespace

} // namespace popx
} // namespace popart
