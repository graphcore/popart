// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/subtract.hpp>
#include <popart/popx/op/subtractx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/op/reducesumx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

SubtractOpx::SubtractOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOpx(op, devicex) {
  verifyOp<SubtractOp>(op, {Onnx::Operators::Sub_6, Onnx::Operators::Sub_7});
}

void SubtractOpx::grow(snap::program::Sequence &prog) const {
  setOutTensor(SubtractOp::getOutIndex(),
               snap::popops::map(graph(),
                                 popops::expr::BinaryOpType::SUBTRACT,
                                 getInTensor(SubtractOp::getArg0InIndex()),
                                 getInTensor(SubtractOp::getArg1InIndex()),
                                 prog,
                                 debugContext()));
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
