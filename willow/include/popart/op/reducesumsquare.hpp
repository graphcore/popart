// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REDUCESUMSQUARE_HPP
#define GUARD_NEURALNET_REDUCESUMSQUARE_HPP

#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp>

namespace popart {

class ReduceSumSquareOp : public ReduceOp {
public:
  ReduceSumSquareOp(const OperatorIdentifier &_opid,
                    const nonstd::optional<std::vector<int64_t>> &axes,
                    const int64_t keepdims,
                    const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;

  void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       CommGroup shardingDomain) final;
};

class ReduceSumSquareGradOp : public ReduceGradOp {
public:
  ReduceSumSquareGradOp(const ReduceSumSquareOp &fwdOp,
                        const Shape &backward_shape);
  std::unique_ptr<Op> clone() const final;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  static InIndex getFwdInInIndex() { return 1; }
};

} // namespace popart

#endif
