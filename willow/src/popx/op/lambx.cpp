// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <ext/new_allocator.h>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/lamb.hpp>
#include <popart/popx/op/lambx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
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

LambSquareOpx::LambSquareOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<LambSquareOp>(op, Onnx::CustomOperators::LambSquare);
}

void LambSquareOpx::grow(poplar::program::Sequence &prog) const {
  auto rsq = popops::reduce(graph(),
                            getInTensor(LambSquareOp::getInIndex()).flatten(),
                            poplar::FLOAT,
                            {0},
                            {popops::Operation::SQUARE_ADD},
                            prog,
                            debugContext("LambSquaredReducedFP32"));

  setOutTensor(LambSquareOp::getOutIndex(), rsq);
}

namespace {
OpxCreator<LambSquareOpx>
    lambSquareOpxCreator(Onnx::CustomOperators::LambSquare);
} // namespace
} // namespace popx
} // namespace popart
