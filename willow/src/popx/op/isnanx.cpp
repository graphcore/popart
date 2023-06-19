// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/isnan.hpp>
#include <popart/popx/op/isnanx.hpp>
#include <popart/popx/opxmanager.hpp>

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

IsNaNx::IsNaNx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<IsNaN>(op, Onnx::Operators::IsNaN_9);
}

void IsNaNx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(IsNaN::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::IS_NAN,
                           getInTensor(IsNaN::getInIndex()),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<IsNaNx> IsNaNxCreator(Onnx::Operators::IsNaN_9);
} // namespace

} // namespace popx
} // namespace popart
