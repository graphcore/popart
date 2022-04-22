// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLREDUCE_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLREDUCE_HPP
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/collectives/collectives.hpp>

#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class Op;
class Tensor;
class TensorInfo;
class ReplicaEqualAnalysisProxy;

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
  std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

private:
  CollectiveOperator op;
};
} // namespace popart

#endif
