// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/log.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/logx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

LogOpx::LogOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<LogOp>(op, Onnx::Operators::Log_6);
}

void LogOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = popops::map(graph(),
                               popops::expr::UnaryOpType::LOGARITHM,
                               getInTensor(LogOp::getInIndex()),
                               prog,
                               debugContext());

  setOutTensor(LogOp::getOutIndex(), outTensor);
}

namespace {
OpxCreator<LogOpx> logOpxCreator(Onnx::Operators::Log_6);
OpxCreator<Opx> logGradOpxCreator(
    Onnx::GradOperators::LogGrad,
    "LogGradOp should be optimised out, \"LogGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
