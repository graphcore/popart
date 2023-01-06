// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/pow.hpp>
#include <popart/popx/op/powx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

PowComputex::PowComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

poplar::Tensor PowComputex::outplace(poplar::program::Sequence &prog,
                                     poplar::Graph &graph,
                                     const poplar::Tensor &a,
                                     const poplar::Tensor &b,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debugStr) const {
  return popops::pow(graph, a, b, prog, {dnai, debugStr});
}

poplar::Tensor PowComputex::maybeInplace(poplar::program::Sequence &prog,
                                         poplar::Graph &graph,
                                         poplar::Tensor &tInOut,
                                         poplar::Tensor &tIn,
                                         const poplar::DebugNameAndId &dnai,
                                         const std::string &name) const {
  return mapMaybeInPlace(graph,
                         popops::expr::BinaryOpType::POWER,
                         tInOut,
                         tIn,
                         prog,
                         {dnai, name},
                         {},
                         name);
}

PowOpx::PowOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<PowComputex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<PowOp>(op, {Onnx::Operators::Pow_1, Onnx::Operators::Pow_7});
}

PowLhsInplaceOpx::PowLhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<PowComputex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<PowLhsInplaceOp>(op);
}

namespace {
OpxCreator<PowOpx> powOpxCreator({Onnx::Operators::Pow_1,
                                  Onnx::Operators::Pow_7});
OpxCreator<PowLhsInplaceOpx>
    powLhsInplaceOpxCreator(Onnx::CustomOperators::PowLhsInplace);
OpxCreator<Opx> powArg0OpxCreator(Onnx::GradOperators::PowArg0Grad,
                                  "PowArg0Grad should be optimised out, "
                                  "\"PowArg0GradOp\" pattern is required");
OpxCreator<Opx> powArg1OpxCreator(Onnx::GradOperators::PowArg1Grad,
                                  "PowArg1Grad should be optimised out, "
                                  "\"PowArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
