// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popops/ElementWise.hpp>
#include <popart/error.hpp>
#include <popart/op/clip.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/clipx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

poplar::Tensor ClipComputex::getClipTensor(float val,
                                           const poplar::Type &type,
                                           poplar::Graph &graph,
                                           const poplar::DebugNameAndId &dnai) {
  auto tensor = graph.addConstant(type, {}, val, {dnai, "clip"});
  graph.setTileMapping(tensor, 0);
  return tensor;
}

poplar::Tensor
ClipComputex::broadcastClipTensor(poplar::Tensor clipT,
                                  const poplar::Tensor &refT) const {
  // Broadcasting clip tensor across each dimension of reference tensor
  auto refShape = refT.shape();

  for (unsigned dim = 0; dim < refShape.size(); ++dim) {
    clipT = clipT.expand({dim});
    clipT = clipT.broadcast(static_cast<uint32_t>(refShape[dim]), dim);
  }
  return clipT;
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
  verifyOp<ClipOp>(op, {Onnx::Operators::Clip_1, Onnx::Operators::Clip_6});
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
  auto clipmax    = ClipComputex::getClipTensor(clipGradOp->getClipMax(),
                                             gradIn.elementType(),
                                             graph(),
                                             getDebugNameAndId());
  auto clipmin    = ClipComputex::getClipTensor(clipGradOp->getClipMin(),
                                             gradIn.elementType(),
                                             graph(),
                                             getDebugNameAndId());

  // 1. Check where clipmin and clipmax are not equal to fwOut
  // 2. Cast as gradin type from bool
  // 3. Multiply 1. and 2.
  // 4. Multiply by gradIn
  // gradin * cast(clipmax != fwdOut) * cast(clipmin != fwdOut)
  auto outTensor = popops::map(
      graph(),
      pe::Mul(
          pe::Mul(pe::_1,
                  pe::Cast(pe::NotEqual(pe::_2, pe::_3), gradIn.elementType())),
          pe::Cast(pe::NotEqual(pe::_2, pe::_4), gradIn.elementType())),
      {gradIn, fwdOut, clipmin, clipmax},
      prog,
      debugPrefix("ApplyMinMaxMask"));

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
