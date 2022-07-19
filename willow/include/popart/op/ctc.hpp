// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_CTC_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_CTC_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/loss.hpp>

#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

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
        const bool zeroInfinity,
        const Op::Settings &settings_,
        const bool enableReducedClassesInLabel,
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
  bool getZeroInfinity() const { return zeroInfinity; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  unsigned getBatchSize() const;
  unsigned getMaxInputLength() const;
  unsigned getMaxTargetLength() const;
  unsigned getNumClasses() const;

  bool canShard() const override { return false; }

  bool getEnableReducedClassesInLabel() const {
    return enableReducedClassesInLabel;
  }

private:
  const unsigned blank;
  const bool zeroInfinity;
  const bool enableReducedClassesInLabel;
  const DataType userOutputType;
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

  bool getEnableReducedClassesInLabel() const {
    return enableReducedClassesInLabel;
  }

private:
  ReductionType reduction;
  TensorInfo logProbsInfo;
  bool enableReducedClassesInLabel;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_CTC_HPP_
