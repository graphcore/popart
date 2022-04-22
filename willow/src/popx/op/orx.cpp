// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/or.hpp>
#include <popart/popx/op/orx.hpp>
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

OrOpx::OrOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<OrOp>(op, {Onnx::Operators::Or_1, Onnx::Operators::Or_7});
}

void OrOpx::grow(snap::program::Sequence &prog) const {

  insert(outId(OrOp::getOutIndex()),
         snap::popops::map(graph(),
                           popops::expr::BinaryOpType::LOGICAL_OR,
                           get(inId(OrOp::getArg0InIndex())),
                           get(inId(OrOp::getArg1InIndex())),
                           prog,
                           debugContext()));
}

namespace {

OpxCreator<OrOpx> orOpxCreator_1(Onnx::Operators::Or_1);
OpxCreator<OrOpx> orOpxCreator_7(Onnx::Operators::Or_7);

} // namespace

} // namespace popx
} // namespace popart
