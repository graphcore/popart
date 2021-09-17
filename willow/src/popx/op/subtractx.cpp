// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/subtract.hpp>
#include <popart/popx/op/subtractx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<SubtractOp>(op, {Onnx::Operators::Sub_6, Onnx::Operators::Sub_7});
}

void SubtractOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(
      SubtractOp::getOutIndex(),
      snap::Tensor{
          popops::map(
              graph().getPoplarGraph(),
              popops::expr::BinaryOpType::SUBTRACT,
              getInTensor(SubtractOp::getArg0InIndex()).getPoplarTensor(),
              getInTensor(SubtractOp::getArg1InIndex()).getPoplarTensor(),
              prog,
              debugContext()),
          graph()});
}

SubtractArg0GradOpx::SubtractArg0GradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  verifyOp<SubtractArg0GradOp>(op, Onnx::GradOperators::SubArg0Grad);
}

namespace {
OpxCreator<SubtractOpx> subtractOpxCreator({Onnx::Operators::Sub_6,
                                            Onnx::Operators::Sub_7});
OpxCreator<SubtractArg0GradOpx>
    subtractArg0GradOpxCreator(Onnx::GradOperators::SubArg0Grad);
} // namespace

} // namespace popx
} // namespace popart
