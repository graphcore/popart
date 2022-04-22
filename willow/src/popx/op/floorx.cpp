// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/popx/op/floorx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace popart {
class FloorInplaceOp;
class FloorOp;
class Op;

namespace popx {
class Devicex;

snap::Tensor FloorComputex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &tensor,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::FLOOR,
                                  tensor.getPoplarTensor(),
                                  prog.getPoplarSequence(),
                                  {dnai, s}),
                      graph};
}

void FloorComputex::inplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  snap::popops::mapInPlace(
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
