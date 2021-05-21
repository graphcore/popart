// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/op/floor.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/floorx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor FloorComputex::outplace(poplar::program::Sequence &prog,
                                       snap::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {

  return popops::map(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::FLOOR,
                     tensor,
                     prog,
                     {dnai, s});
}

void FloorComputex::inplace(poplar::program::Sequence &prog,
                            snap::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::FLOOR,
                     tensor,
                     prog,
                     {dnai, s});
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
