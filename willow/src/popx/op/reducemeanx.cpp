// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <vector>

#include <popart/error.hpp>
#include <popart/op/reducemean.hpp>
#include <popart/popx/op/reducemeanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

#include <snap/popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

ReduceMeanOpx::ReduceMeanOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ReduceMeanOp>(op);
}

void ReduceMeanOpx::grow(snap::program::Sequence &prog) const {
  const auto &op   = getOp<ReduceMeanOp>();
  const auto input = getInTensor(ReduceMeanOp::getInIndex()).getPoplarTensor();

  auto output_tensor =
      snap::Tensor{popops::reduce(graph().getPoplarGraph(),
                                  input,
                                  vector_cast<std::size_t>(op.getAxes()),
                                  {popops::Operation::ADD},
                                  prog.getPoplarSequence(),
                                  debugContext("add")),
                   graph()};

  // TODO: Should this be mapInPlace
  output_tensor = snap::popops::map(
      graph(),
      pe::Divide(pe::_1,
                 pe::Const(inInfo(ReduceMeanOp::getInIndex()).nelms() /
                           outInfo(ReduceMeanOp::getOutIndex()).nelms())),
      {output_tensor},
      prog,
      debugContext("div"));

  setOutTensor(
      ReduceMeanOp::getOutIndex(),
      output_tensor.reshape(outInfo(ReduceMeanOp::getOutIndex()).shape_szt()));
}

ReduceMeanGradOpx::ReduceMeanGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ReduceMeanGradOp>(op, Onnx::GradOperators::ReduceMeanGrad);
}

void ReduceMeanGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &op = getOp<ReduceMeanGradOp>();
  auto output = cloneNcopy(prog, getInTensor(ReduceMeanGradOp::getInIndex()));
  auto input_shape     = inShape(ReduceMeanGradOp::getInIndex());
  auto output_shape    = outShape(ReduceMeanGradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output = snap::popops::map(
      graph(),
      pe::Divide(pe::_1,
                 pe::Const(outInfo(ReduceMeanGradOp::getOutIndex()).nelms() /
                           inInfo(ReduceMeanGradOp::getInIndex()).nelms())),
      {output},
      prog,
      debugContext("div"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceMeanGradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceMeanOpx> reduceMeanOpxCreator(
    {Onnx::Operators::ReduceMean_1, Onnx::Operators::ReduceMean_11});
OpxCreator<ReduceMeanGradOpx>
    reduceMeanGradGradOpxCreator(Onnx::GradOperators::ReduceMeanGrad);
} // namespace

} // namespace popx
} // namespace popart
