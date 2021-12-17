// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDTENSORSHARDINGBUNDLE_HPP
#define GUARD_NEURALNET_REPLICATEDTENSORSHARDINGBUNDLE_HPP

#include <gcl/CollectiveBalancedReorder.hpp>
#include <popart/replicatedtensorsharding.hpp>

namespace popart {
namespace popx {

/**
 * Helper class to bundle all replicated tensor sharding related lowering
 * information together
 */
class ReplicatedTensorShardingBundle {
public:
  /**
   * Construct empty replicated tensor sharding bundle
   * Creates the replicatedTensorShardingTracer with the IR object
   * \param ir IR to create the \c ReplicatedTensorShardingTracer with
   */
  ReplicatedTensorShardingBundle(const Ir &ir);

  /**
   * Check whether a tensor has an associated \c CollectiveBalancedReorder
   * \param tensorId TensorId to check
   * \return         True if the tensor has an associated
   *                 \c CollectiveBalancedReorder
   */
  bool hasCollectiveBalancedReorder(const TensorId &tensorId) const;

  /**
   * Get the associated \c CollectiveBalancedReorder of a tensor.
   * Throws an error if the tensor does not have one.
   * \param tensorId TensorId to return the \c CollectiveBalancedReorder for
   * \return         Shared pointer to the associated
   *                 \c CollectiveBalancedReorder
   */
  std::shared_ptr<gcl::CollectiveBalancedReorder>
  getCollectiveBalancedReorder(const TensorId &tensorId) const;

  /**
   * Get the host rearrangement method of a tensor.
   * Can be applied on the host-side tensor data to rearrange the data before
   * upload or after download to/from the IPU
   * \param tensorId TensorId to return the \c CBR host rearrangement for
   * \return         \c CBR host rearrangement method
   */
  const gcl::CollectiveBalancedHostRearrangement &
  getCollectiveBalancedHostRearrangement(const TensorId &tensorId) const;

  /**
   * Associate an existing \c CollectiveBalancedReorder with a tensor.
   * \param tensorId TensorId to associate the \c CollectiveBalancedReorder with
   * \param cbrId    Identifier of an existing, registered
   *                 \c CollectiveBalancedReorder obtained by
   *                 registerCollectiveBalancedReorder
   */
  void setCollectiveBalancedReorder(const TensorId &tensorId,
                                    CollectiveBalancedReorderId cbrId);

  /**
   * Register a new collective balanced reorder method
   * \param cbr \c GCL \c CollectiveBalancedReoder to register
   * \return    Registered ID for the \c CollectiveBalancedReoder
   */
  CollectiveBalancedReorderId registerCollectiveBalancedReorder(
      std::shared_ptr<gcl::CollectiveBalancedReorder> cbr);

  /**
   *
   * \return
   */
  const std::map<CollectiveBalancedReorderId,
                 std::shared_ptr<gcl::CollectiveBalancedReorder>> &
  getCollectiveReorders() const {
    return collectiveReorders;
  }

  /**
   * \return Tracer to resolve replicated tensor sharding groups
   */
  const ReplicatedTensorShardingTracer &
  getReplicatedTensorShardingTracer() const {
    return replicatedTensorShardingTracer;
  }

  /**
   * \return Tracer to resolve replicated tensor sharding groups
   */
  ReplicatedTensorShardingTracer &getReplicatedTensorShardingTracer() {
    return replicatedTensorShardingTracer;
  }

  /**
   * Get mapping to resolve which \c CollectiveBalancedReorder has to be applied
   * to a tensor to restore the original data order.
   * \return Mapping of all tensors and their associated
   *         \c CollectiveBalancedReorderId
   */
  const std::map<TensorId, CollectiveBalancedReorderId> &
  getCollectiveReorderIds() const {
    return collectiveReorderIds;
  }

private:
  /**
   * Return and increment the CBR counter
   * \return Counter before incrementing that can be used as the next
   *         \c CollectiveBalancedReorderId
   */
  CollectiveBalancedReorderId getAndIncrCBRCounter();

  /**
   * Helper to track replicated tensor sharding related tensors and ops
   */
  ReplicatedTensorShardingTracer replicatedTensorShardingTracer;

  /**
   * Counts the number of registered collectiveReorders
   */
  int cbrCounter;

  /**
   * Map of tensorIds to collective balanced rearrangement IDs for replicated
   * tensor sharded tensors
   */
  std::map<TensorId, CollectiveBalancedReorderId> collectiveReorderIds;

  /**
   * Map of the rearrangement IDs to the rearrangement implementation
   */
  std::map<CollectiveBalancedReorderId,
           std::shared_ptr<gcl::CollectiveBalancedReorder>>
      collectiveReorders;
};

} // namespace popx
} // namespace popart

#endif // GUARD_NEURALNET_IRLOWERING_HPP
