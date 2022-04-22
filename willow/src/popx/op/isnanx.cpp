// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/isnan.hpp>
#include <popart/popx/op/isnanx.hpp>
#include <popart/popx/opxmanager.hpp>

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

IsNaNx::IsNaNx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IsNaN>(op, Onnx::Operators::IsNaN_9);
}

void IsNaNx::grow(snap::program::Sequence &prog) const {
  setOutTensor(IsNaN::getOutIndex(),
               snap::popops::map(graph(),
                                 popops::expr::BinaryOpType::NOT_EQUAL,
                                 getInTensor(IsNaN::getInIndex()),
                                 getInTensor(IsNaN::getInIndex()),
                                 prog,
                                 debugContext()));
}

namespace {
OpxCreator<IsNaNx> IsNaNxCreator(Onnx::Operators::IsNaN_9);
} // namespace

} // namespace popx
} // namespace popart
