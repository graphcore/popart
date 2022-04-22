// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ONNXTOONNX_SPACEDEPTH_HPP
#define GUARD_NEURALNET_ONNXTOONNX_SPACEDEPTH_HPP

#include <cstdint>
#include <memory>
#include <onnxpasses/nodepattern.hpp>
#include <string>
#include <vector>
#include <popart/names.hpp>

#include "onnxpasses/onnxnames.hpp"

namespace popart {
namespace onnxpasses {
class PatternTarget;

// DCR mode: elements along the depth dimension from the input tensor
// are rearranged in the following order: depth, column, and then row.
// CRD mode: elements along the depth dimension from the input tensor
// are rearranged in the following order: column, row, and then depth.
enum class DepthToSpaceMode { DCR, CRD };

// Base class for DepthToSpace and SpaceToDepth.
class SpaceDepth : public NodePattern {
public:
  SpaceDepth(std::shared_ptr<PatternTarget> t) : NodePattern(t) {}

private:
  // Get permutation vector to transpose.
  virtual std::vector<int64_t> perm() const = 0;
  // Get shape to reshape data to higher 6D tensor.
  virtual Shape increaseRank(const Shape &shapeIn, int64_t blockSize) const = 0;
  // Get shape to reshape data to lower 4D tensor.
  virtual Shape decreaseRank(const Shape &shapeIn, int64_t blockSize) const = 0;
  // Get block size.
  virtual int64_t getBlockSize(const NodeProto &node) = 0;
  // Get op type.
  virtual std::string str() const = 0;
  bool go(const NodeProto &node) final;
};

// DepthToSpace rearranges (permutes) data from depth into
// blocks of spatial data. It outputs a copy of the input tensor
// where values from the depth dimension are moved in spatial blocks
// to the height and width dimensions.
// It expect 4 dimensional input tensor: x.shape = b, c, h, w. See
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace

class DepthToSpace : public SpaceDepth {
public:
  DepthToSpace(std::shared_ptr<PatternTarget> t) : SpaceDepth(t) {}

private:
  std::vector<int64_t> perm() const final;
  Shape increaseRank(const Shape &shapeIn, int64_t blockSize) const final;
  Shape decreaseRank(const Shape &shapeIn, int64_t blockSize) const final;
  int64_t getBlockSize(const NodeProto &node) final;
  std::string str() const final { return "DepthToSpace"; }
  void setMode(const DepthToSpaceMode &mode) { depthToSpaceMode = mode; }
  // DepthToSpace mode.
  DepthToSpaceMode depthToSpaceMode;
};

// SpaceToDepth rearranges blocks of spatial data into depth.
// It outputs a copy of the input tensor where values from
// the height and width dimensions are moved to the depth dimension.

class SpaceToDepth : public SpaceDepth {
public:
  SpaceToDepth(std::shared_ptr<PatternTarget> t) : SpaceDepth(t) {}

private:
  std::vector<int64_t> perm() const final;
  Shape increaseRank(const Shape &shapeIn, int64_t blockSize) const final;
  Shape decreaseRank(const Shape &shapeIn, int64_t blockSize) const final;
  int64_t getBlockSize(const NodeProto &node) final;
  std::string str() const final { return "SpaceToDepth"; }
};

} // namespace onnxpasses
} // namespace popart

#endif
