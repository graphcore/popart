// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/equal.hpp>
#include <popart/popx/op/equalx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

EqualOpx::EqualOpx(Op *op, Devicex *devicex)
    : BinaryComparisonOpx(op, devicex) {
  verifyOp<EqualOp>(op,
                    {Onnx::Operators::Equal_1,
                     Onnx::Operators::Equal_7,
                     Onnx::Operators::Equal_11});
}

void EqualOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(EqualOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::BinaryOpType::EQUAL,
                     get(inId(EqualOp::getArg0InIndex())),
                     get(inId(EqualOp::getArg1InIndex())),
                     prog,
                     debugContext()));
}

namespace {

OpxCreator<EqualOpx> equalOpxCreator({Onnx::Operators::Equal_1,
                                      Onnx::Operators::Equal_7,
                                      Onnx::Operators::Equal_11});

} // namespace

} // namespace popx
} // namespace popart
