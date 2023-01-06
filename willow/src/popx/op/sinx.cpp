// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/sin.hpp>
#include <popart/popx/op/sinx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

SinOpx::SinOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SinOp>(op, Onnx::Operators::Sin_7);
}

void SinOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(SinOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::SIN,
                           getInTensor(SinOp::getInIndex()),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<SinOpx> sinOpxCreator(Onnx::Operators::Sin_7);
OpxCreator<Opx> sinGradOpxCreator(
    Onnx::GradOperators::SinGrad,
    "SinGradOp should be optimised out, \"SinGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
