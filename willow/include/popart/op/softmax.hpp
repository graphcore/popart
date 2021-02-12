// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTMAX_HPP
#define GUARD_NEURALNET_SOFTMAX_HPP

#include <popart/op/elementwise.hpp>
#include <popart/op/nll.hpp>

namespace popart {

class SoftmaxOp : public ElementWiseUnaryOp {
public:
  SoftmaxOp(const OperatorIdentifier &_opid,
            int64_t axis_,
            const Op::Settings &);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  int64_t getAxis() const;
  void setAxis(int64_t);

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;

private:
  int64_t axis;
};

class SoftmaxInplaceOp : public ElementWiseInplaceUnaryOp {
public:
  SoftmaxInplaceOp(const SoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  int64_t getAxis() const { return axis; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

private:
  int64_t axis;
};

class SoftmaxGradOp : public Op {
public:
  SoftmaxGradOp(const SoftmaxOp &);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  int gradProbsIn() const;
  int actsIn() const;
  int64_t getAxis() const;

  static InIndex getGradProbsInIndex() { return 0; }
  static InIndex getActsInIndex() { return 1; }
  static OutIndex getOutIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return axis != 0; }

private:
  int64_t axis;
};

class SoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  SoftmaxGradDirectOp(const TensorId lossId,
                      const nonstd::optional<int> ignoreIndex,
                      const ReductionType reduction,
                      const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  bool hasNlllFwdOp() const;
  Op *nlllFwdOp() const;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static InIndex getGradProbsInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction_; }
  bool hasIgnoreIndex() const { return ignoreIndex_ != nonstd::nullopt; }
  nonstd::optional<int> getOptionalIgnoreIndex() const { return ignoreIndex_; }
  int getIgnoreIndex() const { return ignoreIndex_.value(); }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

private:
  TensorId lossId_;
  ReductionType reduction_;
  nonstd::optional<int> ignoreIndex_;
};

class NlllWithSoftmaxGradDirectOp : public Op {
public:
  // where Op in this constructor must be a SoftmaxOp
  // where this is created by a merger between the Op
  // and an NllGradOp
  NlllWithSoftmaxGradDirectOp(const nonstd::optional<int> ignoreIndex,
                              const ReductionType reduction,
                              const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  Op *nlllFwdOp() const;

  static InIndex getProbsInIndex() { return 0; }
  static InIndex getLabelInIndex() { return 1; }
  static InIndex getGradProbsInIndex() { return 2; }

  static OutIndex getLossOutIndex() { return 0; }
  static OutIndex getGradOutIndex() { return 1; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReductionType getReductionType() const { return reduction_; }
  bool hasIgnoreIndex() const { return ignoreIndex_ != nonstd::nullopt; }
  int getIgnoreIndex() const { return ignoreIndex_.value(); }
  virtual void appendOutlineAttributes(OpSerialiserBase &) const final;

  bool canShard() const override { return true; }
  ReductionType getShardReductionType(OutIndex index) const override {
    return getReductionType();
  }
  float getShardRescaleFactor(Op *const shardedOp,
                              OutIndex index) const override;

private:
  ReductionType reduction_;
  nonstd::optional<int> ignoreIndex_;
};

} // namespace popart

#endif
