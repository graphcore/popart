// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/equal.hpp>
#include <popart/popx/op/equalx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

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

void EqualOpx::grow(snap::program::Sequence &prog) const {

  insert(outId(EqualOp::getOutIndex()),
         snap::popops::map(graph(),
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
