// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LAMB_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LAMB_HPP_

#include <memory>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class CommGroup;
class ReplicaGrouping;

class LambSquareOp : public Op {
public:
  LambSquareOp(const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  void setup() final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  bool isOptimizerOp() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;

  // Lamb + replicated tensor sharding:
  // Distributed L2 norm of the weight and updater tensor
  // TODO(T67766): Delete.
  [[deprecated]] void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       CommGroup shardingDomain) final;

  void
  configureForReplicatedTensorSharding(ReplicatedTensorShardingIndices indices,
                                       const ReplicaGrouping &grouping) final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_LAMB_HPP_
