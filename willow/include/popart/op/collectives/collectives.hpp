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
#include "popart/names.hpp"
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
  CollectivesBaseOp(const OperatorIdentifier &_opid,
                    CommGroup group,
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

  void setGCLCommGroup(CommGroup group_) { group = group_; }
  CommGroup getGCLCommGroup() const { return group; }

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
  CommGroup group;
};

class MultiCollectiveBaseOp : public CollectivesBaseOp {
public:
  MultiCollectiveBaseOp(
      const OperatorIdentifier &operatorIdentifier,
      CommGroup commGroup,
      const Op::Settings &settings,
      std::vector<bool> modifiesIndexInplace,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet);
  std::unique_ptr<Op> clone() const override = 0;
  void setup() override;
  VGraphIdAndTileSet getIntrospectionInVirtualGraphId(InIndex in) const;
  VGraphIdAndTileSet getIntrospectionOutVirtualGraphId(OutIndex out) const;
  std::vector<bool> getModifiesIndexInplace() const {
    return modifiesIndexInplace;
  }
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

private:
  std::vector<bool> modifiesIndexInplace;
  std::vector<TensorInfo> outInfoFromBaseOps;
  std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet;
  std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet;
};

/**
 * Extracts \ref CommGroup from op's attributes. If the attribute isn't set,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param attrs Op's attributes.
 * \return \ref CommGroup that is extracted from attributes.
 */
CommGroup extractCommGroupFromAttrs(const Attributes &attrs);

/**
 * Extracts \ref CommGroup from vector of two integers. If the vector is empty,
 * then the function returns a default constructed \ref CommGroup.
 *
 * \param vec Vector of two integers corresponding to the \ref CommGroupType and
 * replicaGroupSize.
 * \return \ref CommGroup that is extracted from the input vector.
 */
CommGroup extractCommGroupFromVector(const std::vector<int64_t> &vec);

/**
 * Calculates the complementary group such that the two CommGroups together
 * span all replicas
 */
CommGroup getComplementCommGroup(const Ir &ir, CommGroup group);

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
CommGroup getComplementCommGroupWithSuperSet(const Ir &ir,
                                             CommGroup group,
                                             CommGroup superSet);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_COLLECTIVES_HPP_
