// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_NLL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_NLL_HPP_

#include <map>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/loss.hpp>
#include <popart/vendored/optional.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class NllOp : public LossOp {
public:
  NllOp(const OperatorIdentifier &_opid,
        const nonstd::optional<int> ignoreIndex,
        const ReductionType reduction,
        bool inputIsLogProbability,
        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  bool hasIgnoreIndex() const { return ignoreIndex_ != nonstd::nullopt; }
  nonstd::optional<int> getOptionalIgnoreIndex() const { return ignoreIndex_; }
  int getIgnoreIndex() const;
  bool inputIsLogProbability() const { return inputIsLogProbability_; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  bool canShard() const override { return true; }
  ReductionType getShardReductionType(OutIndex index) const override {
    return getReductionType();
  }

private:
  // Specifies a target value that is masked when calculating the loss and
  // input gradient
  nonstd::optional<int> ignoreIndex_;

  // Specifies if the input tensor contains log-probabilities
  bool inputIsLogProbability_;
};

class NllGradOp : public Op {
public:
  NllGradOp(const NllOp &);
  NllGradOp(const TensorId &lossId,
            const nonstd::optional<int> ignoreIndex,
            const ReductionType reduction,
            const bool inputIsLogProbability,
            const Op::Settings &settings);

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static InIndex getGradInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction_; }
  bool hasIgnoreIndex() const { return ignoreIndex_ != nonstd::nullopt; }
  nonstd::optional<int> getOptionalIgnoreIndex() const { return ignoreIndex_; }
  int getIgnoreIndex() const;
  bool inputIsLogProbability() const { return inputIsLogProbability_; }
  TensorId getLossTensorId() const { return lossId_; }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  bool canShard() const override { return true; }
  float getShardRescaleFactor(Op *const shardedOp,
                              OutIndex index) const override;

private:
  TensorId lossId_;
  ReductionType reduction_;
  nonstd::optional<int> ignoreIndex_;
  bool inputIsLogProbability_;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_NLL_HPP_
