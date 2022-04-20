// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/roundx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class Op;
class RoundInplaceOp;
class RoundOp;

namespace popx {
class Devicex;

snap::Tensor RoundComputex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &tensor,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::ROUND,
                                  tensor.getPoplarTensor(),
                                  prog.getPoplarSequence(),
                                  {dnai, s}),
                      graph};
}

void RoundComputex::inplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  snap::popops::mapInPlace(
      graph, popops::expr::UnaryOpType::ROUND, tensor, prog, {dnai, s});
}

RoundOpx::RoundOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, RoundComputex::get()) {
  verifyOp<RoundOp>(
      op, {Onnx::Operators::Round_11, Onnx::CustomOperators::Round_1});
}

RoundInplaceOpx::RoundInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, RoundComputex::get()) {
  verifyOp<RoundInplaceOp>(op, Onnx::CustomOperators::RoundInplace);
}

namespace {
OpxCreator<RoundOpx> RoundOpxCreator({Onnx::Operators::Round_11,
                                      Onnx::CustomOperators::Round_1});
OpxCreator<RoundInplaceOpx>
    RoundxInplaceOpxCreator(Onnx::CustomOperators::RoundInplace);
} // namespace

} // namespace popx
} // namespace popart
