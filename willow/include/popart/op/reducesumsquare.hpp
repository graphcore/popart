// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_REDUCESUMSQUARE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_REDUCESUMSQUARE_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/reduce.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/names.hpp"

namespace popart {
class CommGroup;
class ReplicaGrouping;
struct OperatorIdentifier;

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

  // TODO(T67766): Delete.
  [[deprecated]] void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       CommGroup shardingDomain) final;

  void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       const ReplicaGrouping &grouping) final;
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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_REDUCESUMSQUARE_HPP_
