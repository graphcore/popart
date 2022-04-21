// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CTCBEAMSEARCH_HPP
#define GUARD_NEURALNET_CTCBEAMSEARCH_HPP

#include <popart/op.hpp>

namespace popart {
class CtcBeamSearchDecoderOp : public Op {
public:
  CtcBeamSearchDecoderOp(const popart::OperatorIdentifier &_opid,
                         unsigned _blankClass,
                         unsigned _beamWidth,
                         unsigned _topPaths,
                         const popart::Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;

  void setup() final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  float getSubgraphValue() const final;

  bool requiresRandomSeed() const override;

  // Inputs.
  static InIndex getLogProbsInIndex() { return 0; }
  static InIndex getDataLengthsInIndex() { return 1; }

  // Outputs.
  static OutIndex getLabelProbsOutIndex() { return 0; }
  static OutIndex getLabelLengthsOutIndex() { return 1; }
  static OutIndex getDecodedLabelsOutIndex() { return 2; }

  // Attributes.
  unsigned getBlankClass() const { return blankClass; }
  unsigned getBeamWidth() const { return beamWidth; }
  unsigned getTopPaths() const { return topPaths; }
  unsigned getMaxTime() const { return maxTime; }
  unsigned getBatchSize() const { return batchSize; }
  unsigned getNumClasses() const { return numClasses; }

private:
  unsigned blankClass;
  unsigned beamWidth;
  unsigned topPaths;
  unsigned maxTime;
  unsigned batchSize;
  unsigned numClasses;
};
} // namespace popart

#endif // GUARD_NEURALNET_CTCBEAMSEARCH_HPP