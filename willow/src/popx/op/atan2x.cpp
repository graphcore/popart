// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <popart/op/atan2.hpp>
#include <popart/popx/op/atan2x.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/op/elementwisex.hpp"

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

Atan2Computex::Atan2Computex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

snap::Tensor Atan2Computex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &a,
                                     const snap::Tensor &b,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &debugStr) const {
  return snap::popops::atan2(graph, a, b, prog, {dnai, debugStr});
}

snap::Tensor Atan2Computex::maybeInplace(snap::program::Sequence &prog,
                                         snap::Graph &graph,
                                         const snap::Tensor &tInOut,
                                         const snap::Tensor &tIn,
                                         const poplar::DebugNameAndId &dnai,
                                         const std::string &debugStr) const {
  return snap::popops::atan2MaybeInPlace(
      graph, tInOut, tIn, prog, {dnai, debugStr});
}

Atan2Opx::Atan2Opx(Op *op, Devicex *devicex)
    : ElementWiseBinaryOutplaceOpx(
          op,
          devicex,
          std::make_unique<Atan2Computex>(EwbComputex::InplacePolicy::NEVER)) {
  verifyOp<Atan2Op>(op, {Onnx::CustomOperators::Atan2_1});
}

Atan2LhsInplaceOpx::Atan2LhsInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseBinaryInplaceOpx(
          op,
          devicex,
          std::make_unique<Atan2Computex>(EwbComputex::InplacePolicy::LHS)) {
  verifyOp<Atan2LhsInplaceOp>(op);
}

namespace {
OpxCreator<Atan2Opx> atan2OpxCreator({Onnx::CustomOperators::Atan2_1});
OpxCreator<Atan2LhsInplaceOpx>
    atan2LhsInplaceOpxCreator(Onnx::CustomOperators::Atan2Inplace);

} // namespace

} // namespace popx
} // namespace popart
