// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <popops/ExprOp.hpp>
#include <popart/op/less.hpp>
#include <popart/popx/op/lessx.hpp>
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

LessOpx::LessOpx(Op *op, Devicex *devicex) : BinaryComparisonOpx(op, devicex) {
  verifyOp<LessOp>(op, {Onnx::Operators::Less_7, Onnx::Operators::Less_9});
}

void LessOpx::grow(snap::program::Sequence &prog) const {

  insert(outId(LessOp::getOutIndex()),
         snap::popops::map(graph(),
                           popops::expr::BinaryOpType::LESS_THAN,
                           get(inId(LessOp::getArg0InIndex())),
                           get(inId(LessOp::getArg1InIndex())),
                           prog,
                           debugContext()));
}

namespace {

OpxCreator<LessOpx> lessOpxCreator_7(Onnx::Operators::Less_7);
OpxCreator<LessOpx> lessOpxCreator_9(Onnx::Operators::Less_9);

} // namespace

} // namespace popx
} // namespace popart
