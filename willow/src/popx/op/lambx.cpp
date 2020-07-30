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

LambSquareOpx::LambSquareOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LambSquareOp>(op, Onnx::CustomOperators::LambSquare);
}

void LambSquareOpx::grow(poplar::program::Sequence &prog) const {
  auto var = getInTensor(LambSquareOp::getInIndex());

  auto cast = popops::cast(
      graph(), var.flatten(), poplar::FLOAT, prog, debugPrefix("LambCastFP32"));

  auto rsq = popops::reduce(graph(),
                            cast,
                            poplar::FLOAT,
                            {0},
                            {popops::Operation::SQUARE_ADD},
                            prog,
                            debugPrefix("LambSquaredReducedFP32"));

  setOutTensor(LambSquareOp::getOutIndex(), rsq);
}

namespace {
OpxCreator<LambSquareOpx>
    lambSquareOpxCreator(Onnx::CustomOperators::LambSquare);
} // namespace
} // namespace popx
} // namespace popart
