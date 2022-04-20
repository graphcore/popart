// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/and.hpp>
#include <popart/popx/op/andx.hpp>
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

AndOpx::AndOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<AndOp>(op, {Onnx::Operators::And_1, Onnx::Operators::And_7});
}

void AndOpx::grow(snap::program::Sequence &prog) const {

  insert(outId(AndOp::getOutIndex()),
         snap::popops::map(graph(),
                           popops::expr::BinaryOpType::LOGICAL_AND,
                           get(inId(AndOp::getArg0InIndex())),
                           get(inId(AndOp::getArg1InIndex())),
                           prog,
                           debugContext()));
}

namespace {

OpxCreator<AndOpx> andOpxCreator_1(Onnx::Operators::And_1);
OpxCreator<AndOpx> andOpxCreator_7(Onnx::Operators::And_7);

} // namespace

} // namespace popx
} // namespace popart
