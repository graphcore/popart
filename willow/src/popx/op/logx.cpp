// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/log.hpp>
#include <popart/popx/op/logx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

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
