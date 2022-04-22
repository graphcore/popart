// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/log.hpp>
#include <popart/popx/op/logx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

LogOpx::LogOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<LogOp>(op, Onnx::Operators::Log_6);
}

void LogOpx::grow(snap::program::Sequence &prog) const {
  auto outTensor =
      popops::map(graph().getPoplarGraph(),
                  popops::expr::UnaryOpType::LOGARITHM,
                  getInTensor(LogOp::getInIndex()).getPoplarTensor(),
                  prog.getPoplarSequence(),
                  debugContext());

  setOutTensor(LogOp::getOutIndex(), snap::Tensor{outTensor, graph()});
}

namespace {
OpxCreator<LogOpx> logOpxCreator(Onnx::Operators::Log_6);
OpxCreator<PopOpx> logGradOpxCreator(
    Onnx::GradOperators::LogGrad,
    "LogGradOp should be optimised out, \"LogGradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
