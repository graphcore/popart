// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/sin.hpp>
#include <popart/popx/op/sinx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

SinOpx::SinOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SinOp>(op, Onnx::Operators::Sin_7);
}

void SinOpx::grow(poplar::program::Sequence &prog) const {
  setOutTensor(SinOp::getOutIndex(),
               popops::map(graph(),
                           popops::expr::UnaryOpType::SIN,
                           getInTensor(SinOp::getInIndex()),
                           prog,
                           debugContext()));
}

namespace {
OpxCreator<SinOpx> sinOpxCreator(Onnx::Operators::Sin_7);
OpxCreator<Opx> sinGradOpxCreator(
    Onnx::GradOperators::SinGrad,
    "SinGradOp should be optimised out, \"SinGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
