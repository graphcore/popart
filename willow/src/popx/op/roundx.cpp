// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/op/round.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/roundx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor RoundComputex::outplace(poplar::program::Sequence &prog,
                                       snap::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {

  return popops::map(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::ROUND,
                     tensor,
                     prog,
                     {dnai, s});
}

void RoundComputex::inplace(poplar::program::Sequence &prog,
                            snap::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::ROUND,
                     tensor,
                     prog,
                     {dnai, s});
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
OpxCreator<RoundOpx> RoundOpxCreator({Onnx::Operators::Round_11});
OpxCreator<RoundInplaceOpx>
    RoundxInplaceOpxCreator(Onnx::CustomOperators::RoundInplace);
} // namespace

} // namespace popx
} // namespace popart
