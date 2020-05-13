// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NLL_HPP
#define GUARD_NEURALNET_NLL_HPP

#include <popart/op.hpp>
#include <popart/op/loss.hpp>

namespace popart {

class NllOp : public LossOp {
public:
  NllOp(const OperatorIdentifier &_opid,
        const boost::optional<int> ignoreIndex,
        const ReductionType reduction,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction_; }
  bool hasIgnoreIndex() const { return ignoreIndex_ != boost::none; }
  boost::optional<int> getOptionalIgnoreIndex() const { return ignoreIndex_; }
  int getIgnoreIndex() const;
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

private:
  ReductionType reduction_;

  // Specifies a target value that is masked when calculating the loss and
  // input gradient
  boost::optional<int> ignoreIndex_;
};

class NllGradOp : public Op {
public:
  NllGradOp(const NllOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static InIndex getLossScalingInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction_; }
  bool hasIgnoreIndex() const { return ignoreIndex_ != boost::none; }
  boost::optional<int> getOptionalIgnoreIndex() const { return ignoreIndex_; }
  int getIgnoreIndex() const;
  TensorId getLossTensorId() const { return lossId_; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

private:
  TensorId lossId_;
  ReductionType reduction_;
  boost::optional<int> ignoreIndex_;
};

} // namespace popart

#endif
