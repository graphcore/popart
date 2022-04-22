// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/div.hpp>
#include <popart/popx/op/divx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

DivOpx::DivOpx(Op *op, Devicex *devicex) : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<DivOp>(op, {Onnx::Operators::Div_6, Onnx::Operators::Div_7});
}

void DivOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(0,
               snap::popops::map(graph(),
                                 popops::expr::BinaryOpType::DIVIDE,
                                 getInTensor(DivOp::getArg0InIndex()),
                                 getInTensor(DivOp::getArg1InIndex()),
                                 prog,
                                 debugContext()));
}

namespace {
OpxCreator<DivOpx> divOpxCreator({Onnx::Operators::Div_6,
                                  Onnx::Operators::Div_7});
OpxCreator<PopOpx> divArg0OpxCreator(
    Onnx::GradOperators::DivArg0Grad,
    "DivArg0Grad should be optimised out, \"DivArg0Grad\" pattern is required");
OpxCreator<PopOpx> divArg1OpxCreator(Onnx::GradOperators::DivArg1Grad,
                                     "DivArg1Grad should be optimised out, "
                                     "\"DivArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
