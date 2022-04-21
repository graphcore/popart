// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popnn/experimental/ROIAlign.hpp>
#include <popart/op/roialign.hpp>
#include <popart/popx/op/roialignx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

RoiAlignOpx::RoiAlignOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<RoiAlignOp>(op, {Onnx::Operators::RoiAlign_10});
}

void RoiAlignOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor bottomData       = getInTensor(0);
  poplar::Tensor bottomRois       = getInTensor(1);
  poplar::Tensor bottomBatchIndex = getInTensor(2);
  poplar::Tensor topData =
      roiAlignImpl(prog, bottomData, bottomRois, bottomBatchIndex);
  setOutTensor(0, topData);
}

poplar::Tensor
RoiAlignOpx::roiAlignImpl(poplar::program::Sequence &prog,
                          poplar::Tensor &bottomData,
                          poplar::Tensor &bottomRois,
                          poplar::Tensor &bottomBatchIndex) const {
  RoiAlignOp &op = getOp<RoiAlignOp>();

  // Get the input parameters and tensor
  float spatialScale        = op.getSpatialScale();
  uint64_t samplingRatio    = op.getSamplingRatio();
  uint64_t alignedHeight    = op.getAlignedHeight();
  uint64_t alignedWidth     = op.getAlignedWidth();
  const auto roiAlignParams = popnn::experimental::roiAlignParams(
      samplingRatio, alignedHeight, alignedWidth, false, spatialScale);
  poplar::Tensor out = popnn::experimental::roiAlignFwd(graph(),
                                                        prog,
                                                        bottomData,
                                                        bottomRois,
                                                        bottomBatchIndex,
                                                        roiAlignParams,
                                                        "roiAlignFwd");

  return out;
}

RoiAlignGradOpx::RoiAlignGradOpx(Op *op, Devicex *device) : Opx(op, device) {
  verifyOp<RoiAlignGradOp>(op, {Onnx::GradOperators::RoiAlignGrad});
}

void RoiAlignGradOpx::grow(poplar::program::Sequence &prog) const {
  poplar::Tensor topDiff          = getInTensor(0);
  poplar::Tensor bottomRois       = getInTensor(1);
  poplar::Tensor bottomBatchIndex = getInTensor(2);
  poplar::Tensor bottomData       = getInTensor(3);
  poplar::Tensor bottomDiff =
      roiAlignImpl(prog, topDiff, bottomData, bottomRois, bottomBatchIndex);
  setOutTensor(0, bottomDiff);
}

poplar::Tensor
RoiAlignGradOpx::roiAlignImpl(poplar::program::Sequence &prog,
                              poplar::Tensor &topDiff,
                              poplar::Tensor &bottomData,
                              poplar::Tensor &bottomRois,
                              poplar::Tensor &bottomBatchIndex) const {
  RoiAlignGradOp &op = getOp<RoiAlignGradOp>();

  // Get the input parameters and tensor
  float spatialScale        = op.getSpatialScale();
  uint64_t samplingRatio    = op.getSamplingRatio();
  uint64_t alignedHeight    = op.getAlignedHeight();
  uint64_t alignedWidth     = op.getAlignedWidth();
  const auto roiAlignParams = popnn::experimental::roiAlignParams(
      samplingRatio, alignedHeight, alignedWidth, false, spatialScale);
  poplar::Tensor outGrad =
      popnn::experimental::roiAlignInputGradient(graph(),
                                                 prog,
                                                 bottomData,
                                                 bottomRois,
                                                 bottomBatchIndex,
                                                 topDiff,
                                                 roiAlignParams,
                                                 "roiAlignInputGradient");
  return outGrad;
}

namespace {
OpxCreator<RoiAlignOpx> roiAlignOpxCreator({Onnx::Operators::RoiAlign_10});
OpxCreator<RoiAlignGradOpx>
    roiAlignGradOpxCreator(Onnx::GradOperators::RoiAlignGrad);
} // namespace

} // namespace popx
} // namespace popart