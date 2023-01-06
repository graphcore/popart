// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/floorx.hpp>
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
class FloorInplaceOp;
class FloorOp;
class Op;

namespace popx {
class Devicex;

poplar::Tensor FloorComputex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {

  return popops::map(
      graph, popops::expr::UnaryOpType::FLOOR, tensor, prog, {dnai, s});
}

void FloorComputex::inplace(poplar::program::Sequence &prog,
                            poplar::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  popops::mapInPlace(
      graph, popops::expr::UnaryOpType::FLOOR, tensor, prog, {dnai, s});
}

FloorOpx::FloorOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, FloorComputex::get()) {
  verifyOp<FloorOp>(op, {Onnx::Operators::Floor_1, Onnx::Operators::Floor_6});
}

FloorInplaceOpx::FloorInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, FloorComputex::get()) {
  verifyOp<FloorInplaceOp>(op, Onnx::CustomOperators::FloorInplace);
}

namespace {
OpxCreator<FloorOpx> FloorOpxCreator({Onnx::Operators::Floor_1,
                                      Onnx::Operators::Floor_6});
OpxCreator<FloorInplaceOpx>
    floorxInplaceOpxCreator(Onnx::CustomOperators::FloorInplace);
} // namespace

} // namespace popx
} // namespace popart
