// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LAMBOP_HPP
#define GUARD_NEURALNET_LAMBOP_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

class LambSquareOp : public Op {
public:
  LambSquareOp(const Op::Settings &);

  std::unique_ptr<Op> clone() const final;

  void setup() final;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  virtual bool isOptimizerOp() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;

  // Lamb + replicated tensor sharding:
  // Distributed L2 norm of the weight and updater tensor
  void configureForReplicatedTensorSharding(
      ReplicatedTensorShardingIndices indices) final;
};

} // namespace popart

#endif
