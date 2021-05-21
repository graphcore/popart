// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/cos.hpp>
#include <popart/popx/op/cosx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

CosOpx::CosOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<CosOp>(op, Onnx::Operators::Cos_7);
}

void CosOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(CosOp::getOutIndex(),
               popops::map(graph().getPoplarGraph(),
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
