// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <snap/popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/reciprocalx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class ReciprocalOp;

namespace popx {

ReciprocalOpx::ReciprocalOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<ReciprocalOp>(op, Onnx::Operators::Reciprocal_6);
}

void ReciprocalOpx::grow(snap::program::Sequence &prog) const {
  auto ones = getConst(popType(op_p->inInfo(0)), {1}, 1.0, "ones");

  setOutTensor(0,
               snap::popops::map(graph(),
                                 popops::expr::BinaryOpType::DIVIDE,
                                 ones,
                                 getInTensor(0),
                                 prog,
                                 debugContext("divide")));
}

namespace {
OpxCreator<ReciprocalOpx> reciprocalOpxCreator(Onnx::Operators::Reciprocal_6);
} // namespace

} // namespace popx
} // namespace popart
