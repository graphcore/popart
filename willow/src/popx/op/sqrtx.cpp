// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/sqrtx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;
class SqrtOp;

namespace popx {
class Devicex;

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SqrtOp>(op, Onnx::Operators::Sqrt_6);
}

void SqrtOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               popops::map(graph(),
                           popops::expr::UnaryOpType::SQRT,
                           getInTensor(0),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<SqrtOpx> sqrtOpxCreator(Onnx::Operators::Sqrt_6);
} // namespace

} // namespace popx
} // namespace popart
