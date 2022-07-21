// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLGATHER_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLGATHER_HPP
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
  MultiReplicatedAllGatherOp(
      CommGroup commGroup,
      const Settings &settings,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<bool> undoRearrangeForCollective,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);

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

#endif
