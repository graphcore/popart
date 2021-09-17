// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/op/ceil.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/ceilx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

snap::Tensor CeilComputex::outplace(poplar::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {

  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::UnaryOpType::CEIL,
                                  tensor.getPoplarTensor(),
                                  prog,
                                  {dnai, s}),
                      graph};
}

void CeilComputex::inplace(poplar::program::Sequence &prog,
                           snap::Graph &graph,
                           const snap::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::UnaryOpType::CEIL,
                     tensor.getPoplarTensor(),
                     prog,
                     {dnai, s});
}

CeilOpx::CeilOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilOp>(op, {Onnx::Operators::Ceil_1, Onnx::Operators::Ceil_6});
}

CeilInplaceOpx::CeilInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(op, devicex, CeilComputex::get()) {
  verifyOp<CeilInplaceOp>(op, Onnx::CustomOperators::CeilInplace);
}

namespace {
OpxCreator<CeilOpx> ceilOpxCreator({Onnx::Operators::Ceil_1,
                                    Onnx::Operators::Ceil_6});
OpxCreator<CeilInplaceOpx>
    ceilxInplaceOpxCreator(Onnx::CustomOperators::CeilInplace);
} // namespace

} // namespace popx
} // namespace popart
