// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/reducel2.hpp>
#include <popart/popx/op/reducel2x.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

ReduceL2Opx::ReduceL2Opx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReduceL2Op>(op);
}

void ReduceL2Opx::grow(poplar::program::Sequence &prog) const {
  const auto &op   = getOp<ReduceL2Op>();
  const auto input = getInTensor(ReduceL2Op::getInIndex());

  auto output_tensor = popops::reduce(graph(),
                                      input,
                                      vector_cast<std::size_t>(op.getAxes()),
                                      {popops::Operation::SQUARE_ADD},
                                      prog,
                                      debugContext("squareAdd"));
  popops::sqrtInPlace(graph(), output_tensor, prog, debugContext("sqrt"));
  setOutTensor(
      ReduceL2Op::getOutIndex(),
      output_tensor.reshape(outInfo(ReduceL2Op::getOutIndex()).shape_szt()));
}

ReduceL2GradOpx::ReduceL2GradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ReduceL2GradOp>(op, Onnx::GradOperators::ReduceL2Grad);
}

void ReduceL2GradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &op       = getOp<ReduceL2GradOp>();
  auto output          = getInTensor(ReduceL2GradOp::getOutIndex());
  auto scale           = getInTensor(ReduceL2GradOp::getFwdOutInIndex());
  auto fwd_input       = getInTensor(ReduceL2GradOp::getFwdInInIndex());
  auto input_shape     = inShape(ReduceL2GradOp::getInIndex());
  auto output_shape    = outShape(ReduceL2GradOp::getOutIndex());
  const auto new_shape = vector_cast<std::size_t>(op.backwardShape());

  output = output.reshape(new_shape);
  scale  = scale.reshape(new_shape);

  // Broadcasting across each dimension
  for (int dim = 0; dim < new_shape.size(); ++dim) {
    if (new_shape[dim] != output_shape[dim]) {
      output = output.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
      scale  = scale.broadcast(static_cast<uint32_t>(output_shape[dim]), dim);
    }
  }

  output = popops::map(graph(),
                       pe::Divide(pe::Mul(pe::_1, pe::_2), pe::_3),
                       {output, fwd_input, scale},
                       prog,
                       debugContext("output"));

  // output now matches the shape of output_shape
  setOutTensor(ReduceL2GradOp::getOutIndex(), output);
}

namespace {
OpxCreator<ReduceL2Opx> reduceL2OpxCreator({Onnx::Operators::ReduceL2_1,
                                            Onnx::Operators::ReduceL2_11});
OpxCreator<ReduceL2GradOpx>
    reduceL2GradGradOpxCreator(Onnx::GradOperators::ReduceL2Grad);
} // namespace

} // namespace popx
} // namespace popart
