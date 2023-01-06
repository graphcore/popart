// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/roundx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
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
class RoundInplaceOp;
class RoundOp;

namespace popx {
class Devicex;

poplar::Tensor RoundComputex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {

  return popops::map(
      graph, popops::expr::UnaryOpType::ROUND, tensor, prog, {dnai, s});
}

void RoundComputex::inplace(poplar::program::Sequence &prog,
                            poplar::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  popops::mapInPlace(
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
