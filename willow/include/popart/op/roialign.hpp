// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ROIALIGN_HPP
#define GUARD_NEURALNET_ROIALIGN_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

/** Region of Interest (RoI) align operation described in the Mask R-CNN paper.
 *
 * \param spatialScale             Multiplicative spatial scale factor to
 *                                 translate ROI coordinates from their input
 *                                 spatial scale to the scale used when
 *                                 pooling, i.e., spatial scale of the input
 *                                 feature map X relative to the input image.
 * \param samplingRatio            Number of sampling points in the
 *                                 interpolation grid used to compute the
 *                                 output value of each pooled output bin.
 * \param alignedHeight            Pooled output Y's height.
 * \param alignedWidth             Pooled output X's height.
 */
class RoiAlignOp : public Op {
public:
  RoiAlignOp(const popart::OperatorIdentifier &_opid,
             const popart::Op::Settings &settings,
             const float spatialScale,
             const uint64_t samplingRatio,
             const uint64_t alignedHeight,
             const uint64_t alignedWidth);

  RoiAlignOp(const RoiAlignOp &) = default;
  RoiAlignOp &operator=(const RoiAlignOp &) = delete;
  ~RoiAlignOp() override                    = default;
  std::unique_ptr<Op> clone() const final;
  void setup() override;
  std::vector<std::unique_ptr<popart::Op>> getGradOps() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getSpatialScale() const { return spatialScale; }
  uint64_t getSamplingRatio() const { return samplingRatio; }
  uint64_t getAlignedHeight() const { return alignedHeight; }
  uint64_t getAlignedWidth() const { return alignedWidth; }

private:
  float spatialScale;
  uint64_t samplingRatio;
  uint64_t alignedHeight;
  uint64_t alignedWidth;
};

class RoiAlignGradOp : public Op {
public:
  RoiAlignGradOp(const RoiAlignOp &);
  std::unique_ptr<Op> clone() const final;
  virtual void setup();
  virtual const std::vector<popart::GradInOutMapper> &gradInputInfo() const;
  const std::map<int, int> &gradOutToNonGradIn() const;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  float getSpatialScale() const { return spatialScale; }
  uint64_t getSamplingRatio() const { return samplingRatio; }
  uint64_t getAlignedHeight() const { return alignedHeight; }
  uint64_t getAlignedWidth() const { return alignedWidth; }

private:
  float spatialScale;
  uint64_t samplingRatio;
  uint64_t alignedHeight;
  uint64_t alignedWidth;
};

} // namespace popart

#endif
