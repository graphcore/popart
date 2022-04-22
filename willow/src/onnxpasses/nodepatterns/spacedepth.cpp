// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/spacedepth.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <popart/attributes.hpp>
#include <popart/error.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"

namespace popart {
namespace onnxpasses {

namespace {

DepthToSpaceMode getDepthToSpaceModeFromString(const std::string &mode) {

  if (mode == "DCR") {
    return DepthToSpaceMode::DCR;
  } else if (mode == "CRD") {
    return DepthToSpaceMode::CRD;
  } else {
    throw error("Unrecognised DepthToSpaceMode {}", mode);
  }
}

} // namespace

using namespace ONNX_NAMESPACE;

// DepthToSpace:
// We have b, c, h, w = input.shape
// Reshape:
// [b, blocksize, blocksize, c / (blocksize^2), h, w] for DCR
// [b, c / (blocksize ** 2), blocksize, blocksize, h, w]) for CRD
// Transpose to:
// [b, c / (blocksize^2), h, blocksize, w, blocksize]
// Reshape:
// [b, c / blocksize^2, h * blocksize, w * blocksize]

// SpaceToDepth:
// We have N, C, H, W = input.shape
// Reshape:
// [N, C,  H/blocksize, blocksize, W/blocksize, blocksize]
// Transpose to:
// [N, blocksize, blocksize, C, H/blocksize, W/blocksize]
// Reshape:
// [N, C*blocksize^2, H/blocksize, W/blocksize]

bool SpaceDepth::go(const NodeProto &node) {
  if (node.op_type() == str()) {
    int64_t blockSize   = getBlockSize(node);
    const auto inName   = node.input(0);
    const auto outName  = node.output(0);
    const Shape shapeIn = shape(node.input(0)).get();

    // reshape increase rank
    const auto reshapeOutName = withUniqueSuffix(outName);
    auto &nR                  = unary(node, inName, reshapeOutName, "Reshape");
    nR.clear_attribute();
    copyUnderscorePrefixedAttributes(node, nR);
    addIntsAttribute(nR, "shape", increaseRank(shapeIn, blockSize));
    nR.set_domain("ai.graphcore");

    // transpose
    const auto transposeOutName = withUniqueSuffix(outName);
    auto &nT = unary(node, reshapeOutName, transposeOutName, "Transpose");
    nT.clear_attribute();
    copyUnderscorePrefixedAttributes(node, nT);
    addIntsAttribute(nT, "perm", perm());
    nT.set_domain("ai.onnx");

    // reshape decrease rank
    auto &nR2 = unary(node, transposeOutName, outName, "Reshape");
    nR2.clear_attribute();
    copyUnderscorePrefixedAttributes(node, nR2);
    addIntsAttribute(nR2, "shape", decreaseRank(shapeIn, blockSize));
    nR2.set_domain("ai.graphcore");
    return true;
  }
  return false;
}

std::vector<int64_t> DepthToSpace::perm() const {
  // Get permutation vector to transpose to:
  // [b, c / (blocksize^2), h, blocksize, w, blocksize]
  switch (depthToSpaceMode) {
  case DepthToSpaceMode::DCR:
    return {0, 3, 4, 1, 5, 2};
  case DepthToSpaceMode::CRD:
    return {0, 1, 4, 2, 5, 3};
  default:
    throw error("Unrecognised DepthToSpaceMode.");
  }
}

Shape DepthToSpace::decreaseRank(const Shape &shapeIn,
                                 int64_t blockSize) const {
  return {shapeIn[0],
          shapeIn[1] / (blockSize * blockSize),
          shapeIn[2] * blockSize,
          shapeIn[3] * blockSize};
}

Shape DepthToSpace::increaseRank(const Shape &shapeIn,
                                 int64_t blockSize) const {
  // We have b, c, h, w = x.shape
  switch (depthToSpaceMode) {
  case DepthToSpaceMode::DCR:
    // [b, blocksize, blocksize, c / (blocksize^2), h, w]
    return {shapeIn[0],
            blockSize,
            blockSize,
            shapeIn[1] / (blockSize * blockSize),
            shapeIn[2],
            shapeIn[3]};
  case DepthToSpaceMode::CRD:
    // [b, c / (blocksize ** 2), blocksize, blocksize, h, w])
    return {shapeIn[0],
            shapeIn[1] / (blockSize * blockSize),
            blockSize,
            blockSize,
            shapeIn[2],
            shapeIn[3]};
  default:
    throw error("Unrecognised DepthToSpaceMode.");
  }
}

int64_t DepthToSpace::getBlockSize(const NodeProto &node) {
  auto attr      = Attributes(node.attribute());
  auto blockSize = attr.getAttribute<Attributes::Int>("blocksize", 0);
  auto modeStr   = attr.getAttribute<Attributes::String>("mode", "DCR");
  setMode(getDepthToSpaceModeFromString(modeStr));
  return blockSize;
}

std::vector<int64_t> SpaceToDepth::perm() const {
  // Get permutation vector to transpose to:
  // [N, blocksize, blocksize, C, H/blocksize, W/blocksize]
  return {0, 3, 5, 1, 2, 4};
}

Shape SpaceToDepth::decreaseRank(const Shape &shapeIn,
                                 int64_t blockSize) const {
  return {shapeIn[0],
          shapeIn[1] * blockSize * blockSize,
          shapeIn[2] / blockSize,
          shapeIn[3] / blockSize};
}

Shape SpaceToDepth::increaseRank(const Shape &shapeIn,
                                 int64_t blockSize) const {
  // We have N, C, H, W = x.shape
  // to [N, C,  H/blocksize, blocksize, W/blocksize, blocksize]
  return {shapeIn[0],
          shapeIn[1],
          shapeIn[2] / blockSize,
          blockSize,
          shapeIn[3] / blockSize,
          blockSize};
}

int64_t SpaceToDepth::getBlockSize(const NodeProto &node) {
  auto attr      = Attributes(node.attribute());
  auto blockSize = attr.getAttribute<Attributes::Int>("blocksize", 0);
  return blockSize;
}

} // namespace onnxpasses
} // namespace popart
