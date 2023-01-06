// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/not.hpp>
#include <popart/popx/op/notx.hpp>
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

NotOpx::NotOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<NotOp>(op, {Onnx::Operators::Not_1});
}

void NotOpx::grow(poplar::program::Sequence &prog) const {

  insert(outId(NotOp::getOutIndex()),
         popops::map(graph(),
                     popops::expr::UnaryOpType::LOGICAL_NOT,
                     get(inId(NotOp::getInIndex())),
                     prog,
                     debugContext()));
}

namespace {

OpxCreator<NotOpx> notOpxCreator_1(Onnx::Operators::Not_1);

} // namespace

} // namespace popx
} // namespace popart
