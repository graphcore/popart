// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/atan2.hpp>
#include <popart/popx/op/atan2x.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

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

Atan2Computex::Atan2Computex(EwbComputex::InplacePolicy ip) : EwbComputex(ip) {}

poplar::Tensor Atan2Computex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &a,
                                       const poplar::Tensor &b,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &debugStr) const {
  return popops::atan2(graph, a, b, prog, {dnai, debugStr});
}

poplar::Tensor Atan2Computex::maybeInplace(poplar::program::Sequence &prog,
                                           poplar::Graph &graph,
                                           poplar::Tensor &tInOut,
                                           poplar::Tensor &tIn,
                                           const poplar::DebugNameAndId &dnai,
                                           const std::string &name) const {
  return mapMaybeInPlace(graph,
                         popops::expr::BinaryOpType::ATAN2,
                         tInOut,
                         tIn,
                         prog,
                         {dnai, name},
                         {},
                         name);
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
