// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCE_HPP_
#include <memory>
#include <tuple>
#include <vector>
#include <popart/op/collectives/collectives.hpp>

#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class ReplicaGrouping;
class Op;
class Tensor;
class TensorInfo;
class ReplicaEqualAnalysisProxy;

/**
 * A multi-collective class for performing an all-reduce operation on a list of
 *tensors. The tensors will be merged into a single large tensor and reduced as
 *one, leading to better bandwidth utilization and fewer syncs between replicas
 *than doing the all-reduce on a per-tensor basis. The class supports mixing
 *in-place and out-place all-reduce operations, but requires that all tensors
 *use the same collective group i.e. reduction is over the same replicas. This
 *op is usually constructed in the MergeCollectivesTransform
 **/
class MultiReplicatedAllReduceOp : public MultiCollectiveBaseOp {
public:
  /**
   * Constructor for the MultiReplicatedAllReduceOp
   *
   * \param collectiveOperator the collective operator is the same
   * for all input tensors
   * \param commGroup all of the inputs will be reduced across the same
   * communications group
   * \param settings the settings of the op are shared across all inputs
   * \param modifiesIndexInplace for each of the inputs, specify whether
   * it should be modified in place
   * \param outInfoFromBaseOps  the output information for each tensor,
   * usually inherited from a ReplicatedAllReduceOp for that tensor
   * \param inputVirtualGraphIdAndTileSet each input tensor has it's own
   * associated virtual graph
   * \param outputVIrtualGraphIdAnTileSet each output tensor has it's own
   * associated virtual graph
   */
  // TODO(T67766): Delete.
  [[deprecated]] MultiReplicatedAllReduceOp(
      CollectiveOperator collectiveOperator,
      CommGroup commGroup,
      const Settings &settings,
      std::vector<bool> modifiesIndexInplace,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);

  /**
   * Constructor for the MultiReplicatedAllReduceOp
   *
   * \param collectiveOperator the collective operator is the same
   * for all input tensors
   * \param grouping all of the inputs will be reduced across the same
   * communications group
   * \param settings the settings of the op are shared across all inputs
   * \param modifiesIndexInplace for each of the inputs, specify whether
   * it should be modified in place
   * \param outInfoFromBaseOps  the output information for each tensor,
   * usually inherited from a ReplicatedAllReduceOp for that tensor
   * \param inputVirtualGraphIdAndTileSet each input tensor has it's own
   * associated virtual graph
   * \param outputVIrtualGraphIdAnTileSet each output tensor has it's own
   * associated virtual graph
   */
  MultiReplicatedAllReduceOp(
      CollectiveOperator collectiveOperator,
      const ReplicaGrouping &grouping,
      const Settings &settings,
      const std::vector<bool> &modifiesIndexInplace,
      const std::vector<TensorInfo> &outInfoFromBaseOps,
      const std::vector<VGraphIdAndTileSet> &inputVirtualGraphIdAndTileSet,
      const std::vector<VGraphIdAndTileSet> &outputVirtualGraphIdAndTileSet);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  /**
   * Returns the type of the collective used in the all reduce e.g. addition
   * the same collective operator is used across all the inputs to be reduced
   */
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
  /**
   * The collective operation used in the reduce-scatter during lowering
   */
  CollectiveOperator op;
  /**
   * On a per-tensor basis, which inputs should be modified in place
   */
  std::vector<bool> modifiesIndexInplace;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLREDUCE_HPP_
