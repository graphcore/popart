// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/nearbyintx.hpp>
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
class NearbyIntInplaceOp;
class NearbyIntOp;

namespace popx {
class Devicex;

poplar::Tensor NearbyIntComputex::outplace(poplar::program::Sequence &prog,
                                           poplar::Graph &graph,
                                           const poplar::Tensor &tensor,
                                           const poplar::DebugNameAndId &dnai,
                                           const std::string &s) const {

  return popops::map(
      graph, popops::expr::UnaryOpType::NEARBY_INT, tensor, prog, {dnai, s});
}

void NearbyIntComputex::inplace(poplar::program::Sequence &prog,
                                poplar::Graph &graph,
                                const poplar::Tensor &tensor,
                                const poplar::DebugNameAndId &dnai,
                                const std::string &s) const {

  popops::mapInPlace(
      graph, popops::expr::UnaryOpType::NEARBY_INT, tensor, prog, {dnai, s});
}

NearbyIntOpx::NearbyIntOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, NearbyIntComputex::get()) {
  verifyOp<NearbyIntOp>(op, {Onnx::CustomOperators::NearbyInt});
}

NearbyIntInplaceOpx::NearbyIntInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, NearbyIntComputex::get()) {
  verifyOp<NearbyIntInplaceOp>(op, Onnx::CustomOperators::NearbyIntInplace);
}

namespace {
OpxCreator<NearbyIntOpx>
    NearbyIntOpxCreator({Onnx::CustomOperators::NearbyInt});
OpxCreator<NearbyIntInplaceOpx>
    NearbyIntxInplaceOpxCreator(Onnx::CustomOperators::NearbyIntInplace);
} // namespace

} // namespace popx
} // namespace popart
