// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/popx/op/sqrtx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SqrtOpx::SqrtOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SqrtOp>(op, Onnx::Operators::Sqrt_6);
}

void SqrtOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(0,
               snap::Tensor{popops::map(graph().getPoplarGraph(),
                                        popops::expr::UnaryOpType::SQRT,
                                        getInTensor(0).getPoplarTensor(),
                                        prog,
                                        debugContext()),
                            graph()});
}

namespace {
OpxCreator<SqrtOpx> sqrtOpxCreator(Onnx::Operators::Sqrt_6);
} // namespace

} // namespace popx
} // namespace popart
