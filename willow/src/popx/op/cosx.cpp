// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/cos.hpp>
#include <popart/popx/op/cosx.hpp>
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

CosOpx::CosOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<CosOp>(op, Onnx::Operators::Cos_7);
}

void CosOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(CosOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::COS,
                           getInTensor(CosOp::getInIndex()),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<CosOpx> cosOpxCreator(Onnx::Operators::Cos_7);
} // namespace

} // namespace popx
} // namespace popart
