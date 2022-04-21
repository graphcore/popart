// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/lamb.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lambx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/Reduce.hpp>

namespace popart {
namespace popx {

LambSquareOpx::LambSquareOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<LambSquareOp>(op, Onnx::CustomOperators::LambSquare);
}

void LambSquareOpx::grow(snap::program::Sequence &prog) const {
  auto rsq = popops::reduce(
      graph().getPoplarGraph(),
      getInTensor(LambSquareOp::getInIndex()).flatten().getPoplarTensor(),
      poplar::FLOAT,
      {0},
      {popops::Operation::SQUARE_ADD},
      prog.getPoplarSequence(),
      debugContext("LambSquaredReducedFP32"));

  setOutTensor(LambSquareOp::getOutIndex(), snap::Tensor{rsq, graph()});
}

namespace {
OpxCreator<LambSquareOpx>
    lambSquareOpxCreator(Onnx::CustomOperators::LambSquare);
} // namespace
} // namespace popx
} // namespace popart
