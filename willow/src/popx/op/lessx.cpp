// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popart/error.hpp>
#include <popart/op/less.hpp>
#include <popart/popx/devicex.hpp>

#include <popart/popx/op/lessx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace popart {
namespace popx {

LessOpx::LessOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<LessOp>(op, {Onnx::Operators::Less_7, Onnx::Operators::Less_9});
}

void LessOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(LessOp::getOutIndex()),
         snap::Tensor{
             popops::map(graph().getPoplarGraph(),
                         popops::expr::BinaryOpType::LESS_THAN,
                         get(inId(LessOp::getArg0InIndex())).getPoplarTensor(),
                         get(inId(LessOp::getArg1InIndex())).getPoplarTensor(),
                         prog,
                         debugContext()),
             graph()});
}

namespace {

OpxCreator<LessOpx> lessOpxCreator_7(Onnx::Operators::Less_7);
OpxCreator<LessOpx> lessOpxCreator_9(Onnx::Operators::Less_9);

} // namespace

} // namespace popx
} // namespace popart
