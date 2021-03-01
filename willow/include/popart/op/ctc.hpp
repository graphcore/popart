// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CTC_HPP
#define GUARD_NEURALNET_CTC_HPP

#include <popart/op.hpp>
#include <popart/op/loss.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

//
// This op implements the connectionist temporal classification (CTC) loss
// operation.
//
// For CTC, the loss is calculated differently depending on whether we're
// training or doing inference. We currently only support training.
//
// The training loss and gradient are computed with one call to PopLibs. While
// we could call this once in the forward op and once in the backwards op, this
// seems unnecessary wasteful. Instead, we compute both in the forward op and
// pass the gradient to the backward op via an output:
//
//   ,----[logProbs]                             [logProbsGradient]
//   |,---[targets]                                ^
//   ||,--[inputLengths]                           |
//   |||,-[targetLengths]------------------------. |
//   ||||                                         ||
//   vvvv                                         v|
//   CtcOp ----[logProbsGradientWrtCtcLoss]-----> CtcGradOp
//    |                                            ^
//    |                                            |
//    |                                            |
//    v                                            |
//   [ctcLoss]                                   [ctcLossGrad]
//

class CtcOp : public LossOp {
public:
  CtcOp(const OperatorIdentifier &_opid,
        const ReductionType reduction,
        const unsigned blank,
        const Op::Settings &settings_,
        const DataType outDataType = DataType::UNDEFINED);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getLogProbsInIndex() { return 0; }
  static InIndex getTargetsInIndex() { return 1; }
  static InIndex getInputLengthsInIndex() { return 2; }
  static InIndex getTargetLengthsInIndex() { return 3; }

  static OutIndex getCtcLossOutIndex() { return 0; }
  static OutIndex getLogProbsGradientWrtCtcLossOutIndex() { return 1; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  unsigned getBlank() const { return blank; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  unsigned getBatchSize() const { return batchSize; }
  unsigned getMaxInputLength() const { return maxInputLength; }
  unsigned getMaxTargetLength() const { return maxTargetLength; }
  unsigned getNumClasses() const { return numClasses; }

  bool canShard() const override { return false; }

private:
  unsigned blank;
  DataType userOutputType;
  unsigned batchSize;
  unsigned maxInputLength;
  unsigned maxTargetLength;
  unsigned numClasses;
};

class CtcGradOp : public Op {
public:
  CtcGradOp(const CtcOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  // Gradient from fwd op.
  static InIndex getLogProbsGradientWrtCtcLossInIndex() { return 0; }
  // Target lengths.
  static InIndex getTargetLengthsInIndex() { return 1; }
  // Incoming gradient.
  static InIndex getCtcLossGradientInIndex() { return 2; }
  // Gradient output for logarithmized probs.
  static OutIndex getLogProbsGradientOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  bool canShard() const override { return false; }
  ScaleByReplication getScaleByReplication() const {
    return scaleByReplication_;
  }

private:
  ReductionType reduction;
  TensorInfo logProbsInfo;

  // TODO: remove after T34809, as this is now redundant
  const ScaleByReplication scaleByReplication_;
};

} // namespace popart

#endif
