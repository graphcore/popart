// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/scale.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/scalex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

poplar::Tensor ScaleComputex::getScaleTensor(const poplar::Type &type,
                                             poplar::Graph &graph) const {
  auto tensor = graph.addConstant(type, {1}, scale_factor, "scale_factor");
  graph.setTileMapping(tensor, 0);
  return tensor;
}

poplar::Tensor ScaleComputex::outplace(poplar::program::Sequence &prog,
                                       poplar::Graph &graph,
                                       const poplar::Tensor &tensor,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {

  return popops::map(graph,
                     popops::expr::BinaryOpType::MULTIPLY,
                     tensor,
                     getScaleTensor(tensor.elementType(), graph),
                     prog,
                     {dnai, s});
}

float ScaleComputex::getFromScaleOp(Op *op) {
  auto scaleOp = dynamic_cast<ScaleOp *>(op);
  if (scaleOp == nullptr) {
    throw error("Not a valid ScaleOp : {}", op->str());
  }
  return scaleOp->getScaleFactor();
}

float ScaleComputex::getFromScaleInplaceOp(Op *op) {
  auto scaleInOp = dynamic_cast<ScaleInplaceOp *>(op);
  if (scaleInOp == nullptr) {
    throw error("Not a valid ScaleOp : {}", op->str());
  }
  return scaleInOp->getScaleFactor();
}

void ScaleComputex::inplace(poplar::program::Sequence &prog,
                            poplar::Graph &graph,
                            const poplar::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  popops::mapInPlace(graph,
                     popops::expr::BinaryOpType::MULTIPLY,
                     tensor,
                     getScaleTensor(tensor.elementType(), graph),
                     prog,
                     {dnai, s});
}

ScaleOpx::ScaleOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          ScaleComputex::get(ScaleComputex::getFromScaleOp(op))) {}

ScaleInplaceOpx::ScaleInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          ScaleComputex::get(ScaleComputex::getFromScaleInplaceOp(op))) {}

ScaleGradOpx::ScaleGradOpx(Op *op, Devicex *devicex) : ScaleOpx(op, devicex) {
  verifyOp<ScaleGradOp>(op, Onnx::GradOperators::ScaleGrad);
}

namespace {
OpxCreator<ScaleOpx> scaleOpxCreator(Onnx::CustomOperators::Scale_1);
OpxCreator<ScaleInplaceOpx>
    scalexInplaceOpxCreator(Onnx::CustomOperators::ScaleInplace);
OpxCreator<ScaleGradOpx> scaleGradOpxCreator(Onnx::GradOperators::ScaleGrad);
} // namespace

} // namespace popx
} // namespace popart
