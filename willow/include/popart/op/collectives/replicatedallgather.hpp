// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDALLGATHER_HPP
#define GUARD_NEURALNET_REPLICATEDALLGATHER_HPP

#include <popart/op/collectives/collectives.hpp>

namespace popart {

class ReplicatedAllGatherOp : public CollectivesBaseOp {
public:
  ReplicatedAllGatherOp(const OperatorIdentifier &, const Op::Settings &);
  ReplicatedAllGatherOp(const OperatorIdentifier &,
                        const Op::Settings &,
                        TensorInfo outInfo);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

private:
  TensorInfo gatheredOutInfo;
};

} // namespace popart

#endif
