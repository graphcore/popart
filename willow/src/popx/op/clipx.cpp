// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <string>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popart/error.hpp>
#include <popart/op/clip.hpp>
#include <popart/popx/op/clipx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
namespace popx {
class Devicex;

poplar::Tensor ClipComputex::getClipTensor(float val,
                                           const poplar::Type &type,
                                           poplar::Graph &graph,
                                           const poplar::DebugNameAndId &dnai) {
  auto tensor = graph.addConstant(type, {}, val, {dnai, "clip"});
  graph.setTileMapping(tensor, 0);
  return tensor;
}

poplar::Tensor ClipComputex::broadcastClipTensor(poplar::Tensor clipT,
                                                 const poplar::Tensor &refT) {
  // Broadcasting clip tensor across each dimension of reference tensor
  auto refShape = refT.shape();

  auto t = clipT;
  for (unsigned dim = 0; dim < refShape.size(); ++dim) {
    t = t.expand({dim});
    t = t.broadcast(static_cast<uint32_t>(refShape[dim]), dim);
  }
  return t;
}

poplar::Tensor ClipComputex::outplace(poplar::program::Sequence &prog,
                                      poplar::Graph &graph,
                                      const poplar::Tensor &tensor,
                                      const poplar::DebugNameAndId &dnai,
                                      const std::string &s) const {

  auto minT = broadcastClipTensor(
      getClipTensor(min, tensor.elementType(), graph, dnai), tensor);
  auto maxT = broadcastClipTensor(
      getClipTensor(max, tensor.elementType(), graph, dnai), tensor);
  return popops::map(graph,
                     popops::expr::TernaryOpType::CLAMP,
                     tensor,
                     minT,
                     maxT,
                     prog,
                     {dnai, s});
}

ClipOp *ClipComputex::getClipOpFromOp(Op *op) {
  auto clipOp = dynamic_cast<ClipOp *>(op);
  if (clipOp == nullptr) {
    throw error("Not a valid ClipOp : {}", op->str());
  }
  return clipOp;
}

float ClipComputex::getMinFromClipOp(Op *op) {
  auto clipOp = getClipOpFromOp(op);
  return clipOp->getClipMin();
}

float ClipComputex::getMaxFromClipOp(Op *op) {
  auto clipOp = getClipOpFromOp(op);
  return clipOp->getClipMax();
}

ClipInplaceOp *ClipComputex::getClipInplaceOpFromOp(Op *op) {
  auto clipOp = dynamic_cast<ClipInplaceOp *>(op);
  if (clipOp == nullptr) {
    throw error("Not a valid ClipInplaceOp : {}", op->str());
  }
  return clipOp;
}

float ClipComputex::getMinFromClipInplaceOp(Op *op) {
  auto clipInOp = getClipInplaceOpFromOp(op);
  return clipInOp->getClipMin();
}

float ClipComputex::getMaxFromClipInplaceOp(Op *op) {
  auto clipInOp = getClipInplaceOpFromOp(op);
  return clipInOp->getClipMax();
}

void ClipComputex::inplace(poplar::program::Sequence &prog,
                           poplar::Graph &graph,
                           const poplar::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  auto minT = broadcastClipTensor(
      getClipTensor(min, tensor.elementType(), graph, dnai), tensor);
  auto maxT = broadcastClipTensor(
      getClipTensor(max, tensor.elementType(), graph, dnai), tensor);

  popops::mapInPlace(graph,
                     popops::expr::TernaryOpType::CLAMP,
                     tensor,
                     minT,
                     maxT,
                     prog,
                     {dnai, s});
}

ClipOpx::ClipOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          ClipComputex::get(ClipComputex::getMinFromClipOp(op),
                            ClipComputex::getMaxFromClipOp(op))) {
  verifyOp<ClipOp>(op,
                   {Onnx::Operators::Clip_1,
                    Onnx::Operators::Clip_6,
                    Onnx::Operators::Clip_11});
}

ClipInplaceOpx::ClipInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          ClipComputex::get(ClipComputex::getMinFromClipInplaceOp(op),
                            ClipComputex::getMaxFromClipInplaceOp(op))) {
  verifyOp<ClipInplaceOp>(op, Onnx::CustomOperators::ClipInplace);
}

ClipGradOpx::ClipGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ClipGradOp>(op, Onnx::GradOperators::ClipGrad);
}

void ClipGradOpx::grow(poplar::program::Sequence &prog) const {
  // Gradient of clip op is a unit step function, with rising
  // edge at 'min' and falling edge at 'max'

  auto clipGradOp = dynamic_cast<ClipGradOp *>(op_p);
  auto gradIn     = getInTensor(clipGradOp->getGradClippedInIndex());
  auto fwdOut     = getInTensor(clipGradOp->getClippedInIndex());
  auto elType     = gradIn.elementType();
  auto clipmax    = ClipComputex::broadcastClipTensor(
      ClipComputex::getClipTensor(
          clipGradOp->getClipMax(), elType, graph(), getDebugNameAndId()),
      fwdOut);
  auto clipmin = ClipComputex::broadcastClipTensor(
      ClipComputex::getClipTensor(
          clipGradOp->getClipMin(), elType, graph(), getDebugNameAndId()),
      fwdOut);

  // 1. Check where clipmin and clipmax are not equal to fwOut
  // 2. Cast as gradin type from bool
  // 3. Multiply 1. and 2.
  // 4. Multiply by gradIn
  // gradin * cast(clipmax != fwdOut) * cast(clipmin != fwdOut)
  auto outTensor = popops::map(
      graph(),
      pe::Mul(pe::Mul(pe::_1, pe::Cast(pe::NotEqual(pe::_2, pe::_3), elType)),
              pe::Cast(pe::NotEqual(pe::_2, pe::_4), elType)),
      {gradIn, fwdOut, clipmin, clipmax},
      prog,
      debugContext("ApplyMinMaxMask"));

  setOutTensor(clipGradOp->getOutIndex(), outTensor);
}

namespace {
OpxCreator<ClipOpx> clipOpxCreator({Onnx::Operators::Clip_1,
                                    Onnx::Operators::Clip_6,
                                    Onnx::Operators::Clip_11});
OpxCreator<ClipInplaceOpx>
    clipxInplaceOpxCreator(Onnx::CustomOperators::ClipInplace);
OpxCreator<ClipGradOpx> clipGradOpxCreator(Onnx::GradOperators::ClipGrad);
} // namespace

} // namespace popx
} // namespace popart
