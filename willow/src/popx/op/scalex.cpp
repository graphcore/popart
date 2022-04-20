// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <poplar/Graph.hpp>
#include <poplar/Type.hpp>
#include <popops/ExprOp.hpp>
#include <popart/error.hpp>
#include <popart/op/scale.hpp>
#include <popart/popx/op/scalex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
namespace popx {
class Devicex;

snap::Tensor ScaleComputex::getScaleTensor(const poplar::Type &type,
                                           snap::Graph &graph) const {
  auto tensor = graph.getPoplarGraph().addConstant(
      type, {1}, scale_factor, "scale_factor");
  graph.getPoplarGraph().setTileMapping(tensor, 0);
  return snap::Tensor{tensor, graph};
}

snap::Tensor ScaleComputex::outplace(snap::program::Sequence &prog,
                                     snap::Graph &graph,
                                     const snap::Tensor &tensor,
                                     const poplar::DebugNameAndId &dnai,
                                     const std::string &s) const {

  return snap::popops::map(graph,
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

void ScaleComputex::inplace(snap::program::Sequence &prog,
                            snap::Graph &graph,
                            const snap::Tensor &tensor,
                            const poplar::DebugNameAndId &dnai,
                            const std::string &s) const {

  snap::popops::mapInPlace(graph,
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
