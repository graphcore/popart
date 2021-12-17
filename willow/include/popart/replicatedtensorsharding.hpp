// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDTENSORSHARDING_HPP
#define GUARD_NEURALNET_REPLICATEDTENSORSHARDING_HPP

#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>

namespace popart {

using ReplicatedTensorShardingGroupId = int;

/**
 * Struct that describes which inputs/outputs of an Op belong to the
 * sharding group.
 *
 * Regular operations typically belong to only one sharding group, however:
 * - Subgraphing operations (CallOp, LoopOp)
 * - MultiExchangeOp
 * can belong to multiple sharding groups, depending on the input and ouput
 * indices.
 */
struct ReplicatedTensorShardingOpInfo {

  ReplicatedTensorShardingOpInfo() : id(-1), inIndices{}, outIndices{} {}

  ReplicatedTensorShardingOpInfo(OpId id_,
                                 std::set<InIndex> inIndices_,
                                 std::set<OutIndex> outIndices_)
      : id(id_), inIndices(inIndices_), outIndices(outIndices_) {}

  /**
   * Unique ID of the operator
   */
  OpId id;

  /**
   * Input indices belonging to the sharding group
   */
  std::set<InIndex> inIndices;

  /**
   * Output indices belonging to the sharding group
   */
  std::set<OutIndex> outIndices;

  bool operator<(ReplicatedTensorShardingOpInfo const &rhs) const;
};

/**
 * Struct that collects tensors and ops belonging to the same replicated
 * tensor sharding group
 */
struct ReplicatedTensorShardingGroup {

  ReplicatedTensorShardingGroup() : id(-1) {}

  /**
   * Identifier of the group, unique within \c ReplicatedTensorShardingTracer
   */
  ReplicatedTensorShardingGroupId id;

  /**
   * The replicated tensor sharding tensor shape to trace
   * (shape that the sharded tensor currently has)
   */
  nonstd::optional<Shape> shape;

  /**
   * The replicated tensor sharding tensor meta shape to trace
   * (shape that the sharded tensor had before sharding)
   */
  nonstd::optional<Shape> metaShape;

  /**
   * Remote tensors
   */
  std::set<TensorId> remoteTensorIds;

  /**
   * Link tensors
   */
  std::set<TensorId> collectiveLinkedTensorIds;

  /**
   * Sharded tensors
   */
  std::set<TensorId> shardedTensorIds;

  /**
   * Collective operations
   */
  std::map<OpId, ReplicatedTensorShardingOpInfo> collectiveOpIds;

  /**
   * Exchange operations
   */
  std::map<OpId, ReplicatedTensorShardingOpInfo> exchangeOpIds;
};

std::ostream &operator<<(std::ostream &output,
                         const ReplicatedTensorShardingOpInfo &rtsOpId);

std::ostream &operator<<(std::ostream &output,
                         const ReplicatedTensorShardingGroup &rtsGroup);

/**
 * Class that traces the graph and finds all tensors that are:
 * 1.) Replicated tensor sharded
 * 2.) Have the same meta-shape describing the tensor shape before sharding
 * 3.) Use the same collective balanced reorder (CBR) when lowered to Poplar
 * 4.) Share the same elementwise compatible tensor layout by virtue
 *     of 2.) and 3.)
 */
class ReplicatedTensorShardingTracer {
public:
  /**
   * Instantiate the tracer and trace
   * \param ir_           IR to operate on
   * \param startTensors_ Tensors to trace from
   */
  ReplicatedTensorShardingTracer(const Ir &ir_);

  /**
   * Check if the Op associated with the opId has a replicated tensor sharding
   * group
   * \param opInfo \c OpId and input/output indices
   * \return        True if there is a group associated with the \c opId
   */
  bool hasGroup(const ReplicatedTensorShardingOpInfo &opInfo) const;

  /**
   * Check if the tensor associated with the tensorId has a replicated tensor
   * sharding group
   * \param tensorId \c TensorId
   * \return         True if there is a group associated with the \c tensorId
   */
  bool hasGroup(const TensorId &tensorId) const;

  /**
   * Get the replicated tensor sharding group associated with the \c opId
   * \param opInfo \c OpId and input/output indices
   * \return       Associated replicated tensor sharding group.
   */
  const ReplicatedTensorShardingGroup &
  getGroup(const ReplicatedTensorShardingOpInfo &opInfo) const;

  /**
   * Get the replicated tensor sharding group associated with the \c tensorId
   * \param tensorId \c TensorId
   * \return         Associated replicated tensor sharding group.
   */
  const ReplicatedTensorShardingGroup &getGroup(const TensorId &tensorId) const;

  /**
   * Traverse the graph to trace out operators and tensors belonging to the same
   * replicated tensor sharding group.
   * \param startTensors
   */
  void trace(const std::set<Tensor *, PTensorCmp> &startTensors);

private:
  class TraceHelper {
  public:
    /**
     * Visitor applied to each tensor on the path
     * \param t Tensor to visit
     * \return  True If the current path should be traversed
     */
    bool traceVisitor(Tensor *t);

    /**
     * Filter which connections to continue traversal on
     * \param op Operation connecting the tensors tq and tn
     * \param tq Last visited tensor
     * \param tn Next tensor to visit
     * \return   True if the path from tq -> tn should be traversed
     */
    bool traceFilter(Op *op, Tensor *tq, Tensor *tn);

    /**
     * Add a new tensor to start tracing from
     * \param start Tensor to start tracing from
     */
    void addStartTensor(Tensor *start);

    /**
     * Add new tensors to start tracing from
     * \param start Tensors to start tracing from
     */
    void addStartTensors(const std::set<Tensor *, PTensorCmp> &start);

    /**
     * Get the tracing start tensors
     * \return Tracing start tensors
     */
    std::vector<Tensor *> getStartTensors() const;

    /**
     * Capture which tensors are related to which remote buffers
     * \param ir The IR to search for remote buffers
     */
    void registerRemoteBuffers(const Ir &ir);

    /**
     * Capture which remote variables are associated to the group
     * \param ir The IR to search for remote variables
     * \param rbid The remote buffer ID to search associated variables for
     */
    void registerRemoteVariables(const Ir &ir, RemoteBufferId rbid);

    /**
     * Flag to restart traversal with the updated startTensors
     */
    bool restart = false;

    /**
     * Group used during tracing
     */
    ReplicatedTensorShardingGroup group;

  private:
    /**
     * Set of tensors to start from
     */
    std::set<Tensor *> startTensors;

    /**
     * Map from tensors to associated remote buffer IDs
     */
    std::map<Tensor *, std::set<RemoteBufferId>, PTensorCmp>
        tensorRemoteBufferMap;

    /**
     * Map from remote buffer IDs to associated tensors
     */
    std::map<RemoteBufferId, std::set<Tensor *>> remoteBufferTensorMap;
  };

  /**
   * Register a new group
   * \param group Replicated tensor sharding group to register
   */
  void registerGroup(ReplicatedTensorShardingGroup &group);

  /**
   * IR to operate on (all graphs belonging to it can be traversed)
   */
  const Ir &ir;

  /**
   * Vector of structs containing the IDs of operations and tensors belonging to
   * the group. The vector is indexed by \c ReplicatedTensorShardingGroupId.
   */
  std::vector<ReplicatedTensorShardingGroup> groups;

  /**
   * Map of TensorIds to ReplicatedTensorShardingGroupId, used to find the
   * right entry in \c groups.
   */
  std::map<TensorId, ReplicatedTensorShardingGroupId> tensorIdGroupMap;

  /**
   * Map of ReplicatedTensorShardingOpId to ReplicatedTensorShardingGroupId,
   * used to find the right entry in \c groups.
   */
  std::map<ReplicatedTensorShardingOpInfo, ReplicatedTensorShardingGroupId>
      opIdGroupMap;
};

} // namespace popart

#endif
