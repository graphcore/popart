// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLGATHER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLGATHER_HPP_
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
 * A multi-collective class for performing an all-gather operation on a list
 *of tensors. The tensors will be merged into a single large tensor and
 *processed as one, leading to better bandwidth utilization and fewer syncs
 *between replicas than doing the all-gather on a per-tensor basis. All tensors
 *must use the same collective group. This op is usually constructed in the
 *MergeCollectivesTransform.
 **/
class MultiReplicatedAllGatherOp : public MultiCollectiveBaseOp {
public:
  /**
   * Constructor for the MultiReplicatedAllGatherOp
   *
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
  [[deprecated]] MultiReplicatedAllGatherOp(
      CommGroup commGroup,
      const Settings &settings,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<bool> undoRearrangeForCollective,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);

  /**
   * Constructor for the MultiReplicatedAllGatherOp
   *
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
  MultiReplicatedAllGatherOp(
      const ReplicaGrouping &grouping,
      const Settings &settings,
      const std::vector<TensorInfo> &outInfoFromBaseOps,
      const std::vector<bool> &undoRearrangeForCollective,
      const std::vector<VGraphIdAndTileSet> &inputVirtualGraphIdAndTileSet,
      const std::vector<VGraphIdAndTileSet> &outputVirtualGraphIdAndTileSet);

  std::unique_ptr<Op> clone() const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }
  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;
  view::Regions modifies(InIndex index) const override;
  view::Regions aliases(InIndex in, OutIndex out) const override;
  /**
   * Whether the output associated with a given grow part id needs to go through
   * a process of undoing a previously applied rearrangement
   * \param id The Id of the grow part to undo the rearrangement for
   */
  bool undoRearrangeGrowPartForCollective(OpxGrowPartId id) const;

private:
  /**
   * On a per-tensor basis, which outputs should go through an process
   * of undoing a previously applied rearrangement
   */
  std::vector<bool> undoRearrangeForCollective;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_MULTI_REPLICATEDALLGATHER_HPP_
