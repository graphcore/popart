// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_COLLECTIVES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_COLLECTIVES_HPP_

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <set>
#include <vector>
#include <popart/commgroup.hpp>
#include <popart/op.hpp>

#include "popart/attributes.hpp"
#include "popart/ir.hpp"
#include "popart/names.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class Ir;
class OpSerialiserBase;
class Tensor;
struct OperatorIdentifier;

enum class CollectiveOperator {
  Add = 0,
  Mean,
  Mul,
  Min,
  Max,
  LogicalAnd,
  LogicalOr,
  SquareAdd,
  Local,
  N
};

std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op);

class CollectivesBaseOp : public Op {
public:
  // TODO(T67766): Delete.
  [[deprecated]] CollectivesBaseOp(const OperatorIdentifier &_opid,
                                   CommGroup group,
                                   const Op::Settings &settings_);
  CollectivesBaseOp(const OperatorIdentifier &_opid,
                    const ReplicaGrouping &grouping,
                    const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const override = 0;

  // Input to gather/reduce/scatter
  static InIndex getInIndex() { return 0; }

  // Tensor to backtrack collective ops that have to coordinate with each other
  static InIndex getCollectiveLinkedIndex() { return 1; }

  // Gathered/reduced/scattered output
  static OutIndex getOutIndex() { return 0; }

  // Return the default ReplicatedTensorShardingIndicesIndex
  static ReplicatedTensorShardingIndicesIndex
  getDefaultTensorShardingGroupIndex() {
    return 0;
  }

  // Generalization functions for working with linked indices
  virtual bool hasCorrespondingLinkedIndexTensor(Tensor *t);
  bool hasCorrespondingLinkedIndexTensor(InIndex in) {
    return hasCorrespondingLinkedIndexTensor(inTensor(in));
  }
  virtual Tensor *getCorrespondingLinkedIndexTensor(Tensor *t);
  Tensor *getCorrespondingLinkedIndexTensor(InIndex in) {
    return getCorrespondingLinkedIndexTensor(inTensor(in));
  }
  virtual bool isCollectiveLinkedIndexTensor(InIndex in) const;
  virtual bool isCollectiveLinkedIndexTensor(Tensor *t) const;

  // TODO(T67766): Delete.
  [[deprecated]] void setGCLCommGroup(CommGroup group) {
    grouping_ = group.toReplicaGrouping(
        getIr().getSessionOptions().getGlobalReplicationFactor());
  }
  // TODO(T67766): Delete.
  [[deprecated]] CommGroup getGCLCommGroup() const {
    return CommGroup(grouping_);
  }

  void setReplicaGrouping(const ReplicaGrouping &grouping);

  const ReplicaGrouping &getReplicaGrouping() const;

  /**
   * Number of replicas the collective communicates across.
   * This will be used to create a CollectiveBalanceReorder
   * in lowering to improve the tile mapping when using RTS.
   */
  virtual int64_t getCommSize() const;

  void appendOutlineAttributes(OpSerialiserBase &os) const override;

  /**
   * Check \a Replicated tensor sharding (RTS) mode
   * Collective operations setup for RTS are allowed to scramble the data
   * element order of the input (AllGather) / output (ReduceScatter) tensor
   * such that the tensor layouts minimize inter-tile exchanges.
   * As a consequence, the RTS sharded tensor does not follow the original data
   * order and can only be used in elementwise, RTS-enabled operations, such
   * as optimizers, where all inputs consumed are rearranged in the same way.
   * \return True if this operation is configured for replicated tensor sharding
   */
  virtual bool isConfigureOutputForReplicatedTensorSharding() const {
    return false;
  }

private:
  ReplicaGrouping grouping_;
};
/**
 *The base class for a multi-collective which performs all-gather, all-reduce
 *reduce-scatter operations on lists of tensors by first merging them into
 *a larger tensor. This improves bandwidth utilization and decreases the
 *number of syncs needed.
 **/
class MultiCollectiveBaseOp : public CollectivesBaseOp {
public:
  /**
   * Constructor for the MultiReplicatedBaseOp
   *
   * \param operatorIdentifier the identifier for the constructed op
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
  [[deprecated]] MultiCollectiveBaseOp(
      const OperatorIdentifier &operatorIdentifier,
      CommGroup commGroup,
      const Op::Settings &settings,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);
  /**
   * Constructor for the MultiReplicatedBaseOp
   *
   * \param operatorIdentifier the identifier for the constructed op
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
  MultiCollectiveBaseOp(
      const OperatorIdentifier &operatorIdentifier,
      const ReplicaGrouping &grouping,
      const Op::Settings &settings,
      const std::vector<TensorInfo> &outInfoFromBaseOps,
      const std::vector<VGraphIdAndTileSet> &inputVirtualGraphIdAndTileSet,
      const std::vector<VGraphIdAndTileSet> &outputVirtualGraphIdAndTileSet);
  std::unique_ptr<Op> clone() const override = 0;
  void setup() override;
  /**
   * Get virtual graph ID and tile set associated with an input index.
   * \param InIndex The input index.
   * \returns The virtual graph ID and tile set at the input index.
   */
  VGraphIdAndTileSet getIntrospectionInVirtualGraphId(InIndex in) const;
  /**
   * Get virtual graph ID and tile set associated with an output index.
   * \param OutIndex The output index.
   * \returns The virtual graph ID and tile set at the output index.
   */
  VGraphIdAndTileSet getIntrospectionOutVirtualGraphId(OutIndex out) const;
  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex in,
                                   std::set<OpId> &visited) const override;
  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex out,
                                    std::set<OpId> &visited) const override;
  bool hasCorrespondingLinkedIndexTensor(Tensor *t) override;
  Tensor *getCorrespondingLinkedIndexTensor(Tensor *t) override;
  bool isCollectiveLinkedIndexTensor(InIndex in) const override;
  bool isCollectiveLinkedIndexTensor(Tensor *t) const override;
  void growAliasModel(AliasModel &m) const override;

private:
  /**
   * The output information for each tensor, usually inherited from
   * a ReplicatedReduceScatterOp for that tensor. Used to set up the op.
   */
  std::vector<TensorInfo> outInfoFromBaseOps;
  /**
   * Each input tensor has its own associated virtual graph
   */
  std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet;
  /**
   * Each output tensor has its own associated virtual graph
   */
  std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet;
};

/**
 * Extracts \ref CommGroup from op's attributes. If the attribute isn't set,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param attrs Op's attributes.
 * \return \ref CommGroup that is extracted from attributes.
 */
// TODO(T67766): Delete.
[[deprecated]] CommGroup extractCommGroupFromAttrs(const Attributes &attrs);

/**
 * Extracts \ref ReplicaGrouping from op's attributes. If the attribute isn't
 * set, then the function returns a default constructed \ref ReplicaGrouping.
 *
 * \param attrs Op's attributes.
 * \return \ref ReplicaGrouping that is extracted from attributes.
 */
ReplicaGrouping extractReplicaGroupingFromAttrs(const Attributes &attrs,
                                                unsigned replicationFactor);

/**
 * Extracts \ref CommGroup from vector of two integers. If the vector is empty,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param vec Vector of two integers corresponding to the \ref CommGroupType and
 * replicaGroupSize.
 * \return \ref CommGroup that is extracted from the input vector.
 */
// TODO(T67766): Delete.
[[deprecated]] CommGroup
extractCommGroupFromVector(const std::vector<int64_t> &vec);

/**
 * Extracts \ref ReplicaGrouping from vector of three integers. If the vector is
 * empty, then the function returns a default constructed \ref ReplicaGrouping.
 *
 * \param vec Vector of three integers corresponding to the number of replicas,
 *   stride, and group size.
 * \return \ref ReplicaGrouping that is extracted from the input vector.
 */
ReplicaGrouping
extractReplicaGroupingFromVector(const std::vector<int64_t> &vec);

/**
 * Calculates the complementary group such that the two CommGroups together
 * span all replicas
 */
// TODO(T67766): Delete.
[[deprecated]] CommGroup getComplementCommGroup(const Ir &ir, CommGroup group);

/**
 * Calculates the complementary group such that the input group and the
 * complement returned spans all replicas within the superSet. If the superSet
 * is all it derefers to getComplementCommGroup. Will throw error if superSet is
 * not All or if it is equal to the group.
 *
 * \param ir       Handle to get replication factor.
 * \param group    The CommGroup we want the complement of.
 * \param superSet The set to find the complement within.
 * \return         commGroup complement of group within superSet.
 */
// TODO(T67766): Delete.
[[deprecated]] CommGroup getComplementCommGroupWithSuperSet(const Ir &ir,
                                                            CommGroup group,
                                                            CommGroup superSet);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_COLLECTIVES_HPP_
