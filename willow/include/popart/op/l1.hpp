// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_L1_HPP
#define GUARD_NEURALNET_L1_HPP

#include <popart/op.hpp>
#include <popart/op/loss.hpp>

namespace popart {

class L1Op : public LossOp {
public:
  L1Op(const OperatorIdentifier &_opid,
       const float lambda_,
       const ReductionType reduction_,
       const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  float getLambda() const { return lambda; }

  bool canShard() const override { return true; }
  ReductionType getShardReductionType(OutIndex index) const override {
    return getReductionType();
  }

private:
  float lambda;
};

class L1GradOp : public Op {

public:
  L1GradOp(const L1Op &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getFwdActInIndex() { return 0; }
  static InIndex getGradInIndex() { return 1; }

  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  float getLambda() const { return lambda; }
  ReductionType getReductionType() const { return reduction; }

  bool canShard() const override { return true; }
  float getShardRescaleFactor(Op *const shardedOp,
                              OutIndex index) const override;

private:
  float lambda;
  ReductionType reduction;
};

} // namespace popart

#endif
