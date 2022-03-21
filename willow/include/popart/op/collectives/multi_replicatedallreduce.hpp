// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLREDUCE_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLREDUCE_HPP
#include <popart/op/collectives/collectives.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
class MultiReplicatedAllReduceOp : public MultiCollectiveBaseOp {
public:
  MultiReplicatedAllReduceOp(
      CollectiveOperator collectiveOperator,
      CommGroup commGroup,
      const Settings &settings,
      std::vector<bool> modifiesIndexInplace,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  CollectiveOperator getCollectiveOp() const { return op; }
  bool hasCorrespondingLinkedIndexTensor(Tensor *t) override;
  Tensor *getCorrespondingLinkedIndexTensor(Tensor *t) override;
  bool isCollectiveLinkedIndexTensor(InIndex in) const override;
  bool isCollectiveLinkedIndexTensor(Tensor *t) const override;
  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;
  view::Regions modifies(InIndex index) const override;
  view::Regions aliases(InIndex in, OutIndex out) const override;
  void growAliasModel(AliasModel &m) const override;

private:
  CollectiveOperator op;
};
} // namespace popart

#endif