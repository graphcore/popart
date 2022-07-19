// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RESIZE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RESIZE_HPP_

#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
struct OperatorIdentifier;

enum class ResizeMode { Nearest, Linear, Cubic, N };
std::string toString(const ResizeMode &);
std::ostream &operator<<(std::ostream &, const ResizeMode &);

enum class ResizeNearestMode {
  RoundPreferFloor,
  RoundPreferCeil,
  Floor,
  Ceil,
  // Pytorch is not one of the onnx resize modes, but has been added to provide
  // a mode that more closely matches pytorch.interpolate.
  Pytorch,
  N
};

enum class ResizeCoordinateTransformationMode {
  HalfPixel,
  PytorchHalfPixel,
  AlignCorners,
  Asymmetric,
  TfCropAndResize,
  N
};

class ResizeOp : public Op {
public:
  ResizeOp(const OperatorIdentifier &,
           const Op::Settings &,
           ResizeMode,
           const std::vector<float> &scales);

  ResizeOp(const OperatorIdentifier &,
           const Op::Settings &,
           ResizeMode,
           const std::vector<float> &scales,
           ResizeNearestMode nearestMode,
           ResizeCoordinateTransformationMode);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  ResizeMode getMode() const { return mode; }
  const std::vector<float> &getScales() const { return scales; }

  ResizeNearestMode getNearestMode() const { return nearestMode; }
  ResizeCoordinateTransformationMode getCoordinateTransformationMode() const {
    return coordinateTransformationMode;
  }

private:
  const std::vector<float> scales;
  const ResizeMode mode;
  const ResizeNearestMode nearestMode;
  const ResizeCoordinateTransformationMode coordinateTransformationMode;
};

class ResizeGradOp : public ResizeOp {
public:
  ResizeGradOp(const ResizeOp &);

  std::unique_ptr<Op> clone() const override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  const std::vector<float> getFwdScales() const { return fwdScales; }

private:
  const std::vector<float> fwdScales;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RESIZE_HPP_
