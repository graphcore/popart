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

snap::Tensor ClipComputex::getClipTensor(float val,
                                         const poplar::Type &type,
                                         snap::Graph &graph,
                                         const poplar::DebugNameAndId &dnai) {
  auto tensor =
      graph.getPoplarGraph().addConstant(type, {}, val, {dnai, "clip"});
  graph.getPoplarGraph().setTileMapping(tensor, 0);
  return snap::Tensor{tensor, graph};
}

snap::Tensor ClipComputex::broadcastClipTensor(snap::Tensor clipT,
                                               const snap::Tensor &refT) {
  // Broadcasting clip tensor across each dimension of reference tensor
  auto refShape = refT.getPoplarTensor().shape();

  auto t = clipT.getPoplarTensor();
  for (unsigned dim = 0; dim < refShape.size(); ++dim) {
    t = t.expand({dim});
    t = t.broadcast(static_cast<uint32_t>(refShape[dim]), dim);
  }
  return snap::Tensor{t, clipT};
}

snap::Tensor ClipComputex::outplace(poplar::program::Sequence &prog,
                                    snap::Graph &graph,
                                    const snap::Tensor &tensor,
                                    const poplar::DebugNameAndId &dnai,
                                    const std::string &s) const {

  auto minT = broadcastClipTensor(
      getClipTensor(min, tensor.elementType(), graph, dnai), tensor);
  auto maxT = broadcastClipTensor(
      getClipTensor(max, tensor.elementType(), graph, dnai), tensor);
  return snap::Tensor{popops::map(graph.getPoplarGraph(),
                                  popops::expr::TernaryOpType::CLAMP,
                                  tensor.getPoplarTensor(),
                                  minT.getPoplarTensor(),
                                  maxT.getPoplarTensor(),
                                  prog,
                                  {dnai, s}),
                      graph};
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
                           snap::Graph &graph,
                           const snap::Tensor &tensor,
                           const poplar::DebugNameAndId &dnai,
                           const std::string &s) const {

  auto minT = broadcastClipTensor(
      getClipTensor(min, tensor.elementType(), graph, dnai), tensor);
  auto maxT = broadcastClipTensor(
      getClipTensor(max, tensor.elementType(), graph, dnai), tensor);

  popops::mapInPlace(graph.getPoplarGraph(),
                     popops::expr::TernaryOpType::CLAMP,
                     tensor.getPoplarTensor(),
                     minT.getPoplarTensor(),
                     maxT.getPoplarTensor(),
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

ClipGradOpx::ClipGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
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
  auto outTensor = snap::Tensor{
      popops::map(
          graph().getPoplarGraph(),
          pe::Mul(
              pe::Mul(pe::_1, pe::Cast(pe::NotEqual(pe::_2, pe::_3), elType)),
              pe::Cast(pe::NotEqual(pe::_2, pe::_4), elType)),
          {gradIn.getPoplarTensor(),
           fwdOut.getPoplarTensor(),
           clipmin.getPoplarTensor(),
           clipmax.getPoplarTensor()},
          prog,
          debugContext("ApplyMinMaxMask")),
      graph()};

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
