// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/div.hpp>
#include <popart/popx/op/divx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <snap/popops/ElementWise.hpp>

namespace popart {
namespace popx {

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
