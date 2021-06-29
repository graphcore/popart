// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCHEDULE_ACCUMULATEOUTERFRAGMENTPARALLELIZER_HPP
#define GUARD_NEURALNET_SCHEDULE_ACCUMULATEOUTERFRAGMENTPARALLELIZER_HPP

#include <popart/graph.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

#include <list>
#include <set>
#include <vector>

namespace popart {

class AccumulateOuterFragmentParallelizer : public Transform {
public:
  static std::size_t id();

  AccumulateOuterFragmentParallelizer();
  virtual ~AccumulateOuterFragmentParallelizer();

  // We add topological constraints between remote ops as a graph tranform
  // to try and get them adjacent in the scheduler.
  virtual bool apply(Graph &graph) const final;

  // We provide bin constraints that can additionally be applied by the
  // scheduler to speed up scheduling.
  virtual std::vector<std::vector<Op *>>
  getBinConstraints(const Graph &graph) const;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final {
    return "AccumuateOuterFragmentParallelizer";
  }

protected:
  // A class representing a group of operations in the accumulator outer
  // fragment. that logically go together. Each group of operations is
  // independent from one another, no shared tensors. We use hashes over tensor
  // IDs to check this quickly.
  class OpCluster {
  public:
    using Ops       = std::vector<Op *>;
    using TensorIds = std::vector<TensorId>;

    OpCluster(const Graph *graph, Op *op);
    OpCluster()                     = delete;
    OpCluster(const OpCluster &rhs) = default;

    // True if we know OpClusters to use the same tensors.
    bool overlaps(const OpCluster &rhs) const;
    // True if there are *any* remote ops here.
    bool hasRemoteOp() const;
    // Add OpCluster to ours.
    void absorb(const OpCluster &rhs);
    // True if there is a virtual graph id all ops agree on.
    bool hasVirtualGraphId() const;
    // Get the virtual graph id if there is one.
    VGraphId getVirtualGraphId() const;
    // Get the number of load bytes (if split over multiple virtual graph ids,
    // pick max)
    int64_t calcNumLoadBytes() const;

    // Get the tensor IDs for op's inputs and outputs and put them in the
    // tensorIds set.
    static void gatherTensorIds(const Op *op, TensorIds &tensorIds);

    // The graph.
    const Graph *graph;
    // The ops that form part of this group (sorted for quick intersection
    // check).
    Ops ops;
    // Topologically adjacent ops (sorted for quick intersection check).
    Ops adjacentOps;
    // The ops of type RemoteLoadOp.
    Ops remoteLoadOps;
    // The ops of type RemoteStoreOp.
    Ops remoteStoreOps;
    // The ops of type MultiExchangeOp.
    Ops multiExchangeOps;
    // The sum of the number of elements of tensors being loaded.
    int64_t numLoadBytes;
    std::set<Shape> loadShapes;
    // VGraphId as shared by the ops in this group.
    VGraphId virtualGraphId;
    // The tensors IDs that are inputs/outputs of ops in this group. This set is
    // used to detect overlaps between groups with reasonable efficiency. This
    // vector is sorted so overlap checks are quick.
    TensorIds tensorIds;
  };

  using Ops           = std::vector<Op *>;
  using OpClusters    = std::list<OpCluster>;
  using OpClustersMap = std::map<VGraphId, OpClusters>;

  // Get a partitioning of ops in the accumulate outer fragment.
  void populateOpClusters(const Graph &graph, OpClusters &clusters) const;
  void populateOpClusters(const Graph &graph, OpClustersMap &map) const;
  // Filter out clusters without remote ops.
  void filterOpClusters(OpClusters &clusters) const;
  void filterOpClusters(OpClustersMap &map) const;
  // Sort clusters in descending number-of-loaded-bytes size.
  void sortOpClusters(OpClusters &clusters) const;
  void sortOpClusters(OpClustersMap &map) const;
  // Try and parallelise the op clusters. Note we pass a vector of iterators
  // to OpCluster objects in an attempt to save an unnecessary copy.
  void tryToParallelizeOpClusters(Graph &graph, OpClusters &opClusters) const;
  // Helper function to add constraints for group of ops.
  void addOpConstraints(Graph &graph, const Ops &ops) const;
};

} // namespace popart

#endif
