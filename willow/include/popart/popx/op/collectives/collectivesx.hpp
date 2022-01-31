// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVESX_HPP
#define GUARD_NEURALNET_COLLECTIVESX_HPP

#include <popart/debugcontext.hpp>
#include <popart/names.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/popx/viewchangers.hpp>
#include <popart/replicatedtensorsharding.hpp>

#include <popops/CollectiveTypes.hpp>

#include <gcl/CollectiveBalancedReorder.hpp>
#include <gcl/Collectives.hpp>

namespace popart {
namespace popx {

gcl::CollectiveOperator getPoplarCollectiveOperator(CollectiveOperator op);

struct ReorderMetadata {
  ReorderMetadata(int64_t offset_,
                  int64_t rearranged_offset_,
                  int64_t size_,
                  int64_t tile_)
      : offset(offset_), rearranged_offset(rearranged_offset_), size(size_),
        tile(tile_) {}
  int64_t offset;
  int64_t rearranged_offset;
  int64_t size;
  int64_t tile;
};

// If the input/output to a collective op is padded,
// provide a cut-down tensor that fits IR specifications of that tensor
// Note: The cut-down tensor may not contain all regions of the tensor that are
// carrying data, and is therefore not suited to be operated on and only
// serves to provide an IR-compatible view of the tensor
class ReplicatedGatherInScatterOutViewChanger : public ViewChanger {
public:
  ReplicatedGatherInScatterOutViewChanger(
      int64_t nelms_,
      ReplicatedTensorShardingGroupId group_)
      : nelms(nelms_), group(group_) {}
  snap::Tensor apply(snap::Tensor tensor) const final {
    return tensor.slice(0, nelms, 0);
  }
  bool containsAllDataRegions() const final { return false; }
  bool operator==(const ViewChanger &rhs) const final {
    if (const ReplicatedGatherInScatterOutViewChanger *other =
            dynamic_cast<const ReplicatedGatherInScatterOutViewChanger *>(
                &rhs)) {
      return group == other->group;
    }
    return false;
  }

private:
  int64_t nelms;
  ReplicatedTensorShardingGroupId group;
};

// If the (tile-balanced) input/output to a collective op is rearranged and/or
// padded, provide a view of the tenosr data that matches the IR expectations
// of the tensor.
// Note: The view is suitable for consumption by all ops, as all data carrying
// regions are included in the view and arranged correctly
class ReplicatedGatherOutScatterInViewChanger : public ViewChanger {
public:
  ReplicatedGatherOutScatterInViewChanger(
      const gcl::CollectiveBalancedReorder *cbr_,
      ReplicatedTensorShardingGroupId group_)
      : cbr(cbr_), group(group_) {}
  snap::Tensor apply(snap::Tensor tensor) const final {
    return snap::Tensor{
        cbr->undoRearrangeForCollective(tensor.getPoplarTensor())
            .reshape(cbr->getReferenceShape()),
        tensor};
  }
  bool operator==(const ViewChanger &rhs) const final {
    if (const ReplicatedGatherOutScatterInViewChanger *other =
            dynamic_cast<const ReplicatedGatherOutScatterInViewChanger *>(
                &rhs)) {
      return group == other->group;
    }
    return false;
  }

private:
  const gcl::CollectiveBalancedReorder *cbr;
  ReplicatedTensorShardingGroupId group;
};

class CollectivesBaseOpx : public PopOpx {
public:
  CollectivesBaseOpx(Op *, Devicex *);

  /**
   * Function to determine which \a collective Ops need to be in the same
   * collective linked group.
   *
   * Ops in the same \a collective linked group need to use the same
   * collective balanced reorder to ensure tensor layouts of tensors that
   * interact with each other in the graph, are compatible.
   *
   *  Scenarios leading to \a collective Ops belonging to the same group:
   *
   *  1. The \c CollectivesBaseOp::getCollectiveLinkedIndex()
   *     is connected to the same root tensor (i.e. tensor A connects to the
   *     \c getCollectiveLinkedIndex of a \c ReduceScatter and \c AllGather,
   *     directly or indirectly):
   *
   *     A -> ReduceScatter
   *      \-> IdentiyOp -> AllGather
   *
   *  2. The \a RTS enabled input/output tensors of \a RTS enabled collective
   *     operations meet in the compute graph:
   *
   *     B -> ReduceScatter -> C -> AllGather -> F -> ReduceScatter -> G
   *                            \
   *                             VarUpdateOp
   *                            /
   *     D -> ReduceScatter -> E
   *
   *     C, E and the VarUpdateOp in this graph are replicated tensor sharded
   *     (\a RTS) and therefore, both ReduceScatter Ops and the AllGather Op
   *     end up in the same \a collective linked group.
   *     B, D, F, G are not sharded, and therefore, the ReduceScatter between
   *     F and G can be in a different \a collective linked group.
   *
   * The primary motivation for \a collective linked groups is "folding"
   * multiple RTS tensors together via e.g. \a outlining.
   * Folding in this context is when two operations or tensors that were unique
   * now use the same code or memory, which implies that for example tensor
   * layouts need to be identical too.
   * If the graph has 3 \a RTS enabled variables, for example, and 2 of them
   * use the same \c VarUpdateOp due to outlining, this implies that we need to
   * ensure all \a RTS related Ops connected to those 2 variables use identical
   * \a CBR (collective balanced reorder) rearrangement.
   *
   * \a CBR is set in the collective Ops themselves either during
   * \c Opx::unwindTensorLayout, \c Opx:createInputTensor or \c Opx::grow
   * by calling \c createCollectiveBalancedReorder
   *
   * The third variable would use a separate VarUpdateOp, and therefore is in a
   * separate \a collective linked group, and can instantiate it's own \a CBR,
   * even if the tensor shapes matches.
   *
   * \c getCollectiveLinkedGroup uses Ops that introduce \a RTS/CBR as a
   * starting point (\c ReduceScatter & \c AllGather) and tracks all associated
   * Ops that propagate \a RTS with a \a DFS search on the graph.
   *
   * \return  Returns all linked tensors and their connected ops to coordinate
   *          tensor mapping of collective inputs and outputs
   */
  ReplicatedTensorShardingGroup getCollectiveLinkedGroup() const;

  /**
   * Get the existing \a CBR
   * \return Existing CBR for the input/output tensor of the collective Op
   */
  gcl::CollectiveBalancedReorder *getCollectiveBalancedReorder() const;

  /**
   * Create a new \a CBR instance for the reference \c tensor
   * \param tensor non-sharded reference tensor
   * \return New CBR for the input/output tensor of the collective Op
   */
  gcl::CollectiveBalancedReorder *
  createCollectiveBalancedReorder(snap::Tensor tensor) const;
};

/**
 * Converts give \ref CommGroup to GCL's CommGroup type.

 * \param input PopART \ref CommGroup.
 * \return GCL CommGroup.
 */
gcl::CommGroup toGCLCommGroup(const popart::CommGroup &input);

} // namespace popx
} // namespace popart

#endif
