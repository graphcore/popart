// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <popart/op/pow.hpp>
#include <popart/popx/op/powx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
class Graph;
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

PowComputex::PowComputex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

snap::Tensor PowComputex::outplace(snap::program::Sequence &prog,
                                   snap::Graph &graph,
                                   const snap::Tensor &a,
                                   const snap::Tensor &b,
                                   const poplar::DebugNameAndId &dnai,
                                   const std::string &debugStr) const {
  return snap::popops::pow(graph, a, b, prog, {dnai, debugStr});
}

snap::Tensor PowComputex::maybeInplace(snap::program::Sequence &prog,
                                       snap::Graph &graph,
                                       const snap::Tensor &tInOut,
                                       const snap::Tensor &tIn,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &debugStr) const {
  return snap::popops::powMaybeInPlace(
      graph, tInOut, tIn, prog, {dnai, debugStr});
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
OpxCreator<PopOpx> powArg0OpxCreator(Onnx::GradOperators::PowArg0Grad,
                                     "PowArg0Grad should be optimised out, "
                                     "\"PowArg0GradOp\" pattern is required");
OpxCreator<PopOpx> powArg1OpxCreator(Onnx::GradOperators::PowArg1Grad,
                                     "PowArg1Grad should be optimised out, "
                                     "\"PowArg1GradOp\" pattern is required");
} // namespace

} // namespace popx
} // namespace popart
