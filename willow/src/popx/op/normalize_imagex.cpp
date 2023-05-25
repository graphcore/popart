// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <iosfwd>
#include <vector>

#include <popart/graphcoreoperators.hpp>
#include <popart/op/normalize_image.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/op/normalize_imagex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/region.hpp" // IWYU pragma: keep
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/tensorinfo.hpp>

#include <poplar/ArrayRef.hpp>
#include <popops/NormaliseImage.hpp>

namespace poplar {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

namespace popx {

// poplar implementation
NormalizeImageOpx::NormalizeImageOpx(popart::Op *op,
                                     popart::popx::Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<NormalizeImageOp>(op, {Onnx::CustomOperators::NormalizeImageOpId});
}

poplar::Tensor
NormalizeImageOpx::createInput(popart::InIndex index,
                               const poplar::DebugNameAndId &dnai) const {
  if (index == NormalizeImageOp::getImageInIndex()) {
    poplar::Tensor imgSrc = createNormalizedImageInput(dnai);
    return imgSrc;
  }
  return Opx::createInput(index, dnai);
}

popart::popx::InputCreatorType
NormalizeImageOpx::getInputCreatorType(popart::InIndex index) const {
  if (index == NormalizeImageOp::getImageInIndex()) {
    return popart::popx::InputCreatorType::CanCreate;
  }
  return popart::popx::InputCreatorType::Deadend;
}

std::set<popart::TensorId>
NormalizeImageOpx::mustExistBeforeCreate(popart::InIndex) const {
  return {};
}

poplar::Tensor NormalizeImageOpx::createNormalizedImageInput(
    const poplar::DebugNameAndId &dnai) const {
  auto inImgInfo = inInfo(NormalizeImageOp::getImageInIndex());
  auto &g        = graph();
  auto raw_shape = inImgInfo.shape_szt();
  const poplar::ArrayRef<size_t> shape(raw_shape.data(), raw_shape.size());
  poplar::Tensor imgSrc = popops::createNormaliseImageInput(
      g, popart::popx::popType(inImgInfo), shape, dnai);
  return imgSrc;
}

void NormalizeImageOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<NormalizeImageOp>();
  auto &g  = graph();

  const float scale = op.getScale();

  poplar::Tensor imgRef = getInTensor(NormalizeImageOp::getImageInIndex());

  poplar::Tensor offsets = getInTensor(NormalizeImageOp::getOffsetsInIndex());

  poplar::Tensor scales = getInTensor(NormalizeImageOp::getScalesInIndex());

  poplar::Tensor out =
      popops::normaliseImage(g,
                             prog,
                             imgRef,
                             scale,
                             offsets,
                             scales,
                             {NormalizeImageOp::opName() + "/normalize"});

  setOutTensor(NormalizeImageOp::getOutIndex(), out);
}

namespace {
popart::popx::OpxCreator<NormalizeImageOpx>
    normalizeImageOpxCreator({Onnx::CustomOperators::NormalizeImageOpId});
} // namespace

} // namespace popx
} // namespace popart
