// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDREDUCESCATTER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDREDUCESCATTER_HPP_
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
 * A multi-collective class for performing a reduce-scatter operation on a list
 *of tensors. The tensors will be merged into a single large tensor and
 *processed as one, leading to better bandwidth utilization and fewer syncs
 *between replicas than doing the reduce-scatter on a per-tensor basis. All
 *tensors must use the same collective group i.e. reduction is over the same
 *replicas. This op is usually constructed in the MergeCollectivesTransform
 **/
class MultiReplicatedReduceScatterOp : public MultiCollectiveBaseOp {
public:
  /**
   * Constructor for the MultiReplicatedReduceScatterOp
   *
   * \param collectiveOperator the collective operator is the same
   * for all input tensors
   * \param commGroup all of the inputs will be reduced scattered across
   * the same communications group
   * \param settings the settings of the op are shared across all inputs
   * \param outInfoFromBaseOps the output information for each tensor,
   * usually inherited from a ReplicatedReduceScatterOp for that tensor
   * \param inputVirtualGraphIdAndTileSet each input tensor has it's own
   * associated virtual graph
   * \param outputVIrtualGraphIdAnTileSet each output tensor has it's own
   * associated virtual graph
   */
  // TODO(T67766): Delete.
  [[deprecated]] MultiReplicatedReduceScatterOp(
      CollectiveOperator collectiveOperator,
      CommGroup commGroup,
      const Settings &settings,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<bool> rearrangeForCollective,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);

  /**
   * Constructor for the MultiReplicatedReduceScatterOp
   *
   * \param collectiveOperator the collective operator is the same
   * for all input tensors
   * \param grouping all of the inputs will be reduced scattered across
   * the same communications group
   * \param settings the settings of the op are shared across all inputs
   * \param outInfoFromBaseOps the output information for each tensor,
   * usually inherited from a ReplicatedReduceScatterOp for that tensor
   * \param inputVirtualGraphIdAndTileSet each input tensor has it's own
   * associated virtual graph
   * \param outputVIrtualGraphIdAnTileSet each output tensor has it's own
   * associated virtual graph
   */
  MultiReplicatedReduceScatterOp(
      CollectiveOperator collectiveOperator,
      const ReplicaGrouping &grouping,
      const Settings &settings,
      const std::vector<TensorInfo> &outInfoFromBaseOps,
      const std::vector<bool> &rearrangeForCollective,
      const std::vector<VGraphIdAndTileSet> &inputVirtualGraphIdAndTileSet,
      const std::vector<VGraphIdAndTileSet> &outputVirtualGraphIdAndTileSet);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  /**
   * Returns the type of the collective used in the all reduce e.g. addition
   * the same collective operator is used across all the inputs to be reduced
   */
  CollectiveOperator getCollectiveOp() const { return op; }
  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;
  view::Regions modifies(InIndex index) const override;
  view::Regions aliases(InIndex in, OutIndex out) const override;
  /**
   * Whether the input of a given grow part needs to be rearranged before the
   * collective operation. Used while lower the op.
   * \param id The Id of the grow part where the input may need re-arranging
   */
  bool rearrangeGrowPartForCollective(OpxGrowPartId id) const;

private:
  /**
   * The collective operation used in the reduce-scatter during lowering
   */
  CollectiveOperator op;
  /**
   * On a per-tensor basis, which inputs should be rearranged before the
   * collective operation
   */
  std::vector<bool> rearrangeForCollective;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDREDUCESCATTER_HPP_
