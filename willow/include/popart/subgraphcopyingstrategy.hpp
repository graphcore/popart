// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPH_COPYING_STRATEGY_HPP
#define GUARD_NEURALNET_SUBGRAPH_COPYING_STRATEGY_HPP

#include <deque>
#include <iostream>
#include <map>
#include <vector>

#include <popart/aliasesmap.hpp>
#include <popart/liveness.hpp>
#include <popart/names.hpp>

namespace popart {

class Ir;
class Op;
class CallOp;

namespace liveness {

/**
 * A base class to define a strategy to determine the scheduling of copies for
 * subgraphs. See subgraphpartitioner.hpp for a motivation as to why we want
 * sometimes partition subgraphs.
 *
 * This class is an interface used by LivenessAnalyzer to determine where to
 * place input/output copies for subgraphs which, in turn, is used by
 * the SubgraphPartitioner to determine how to partition subgraphs.
 *
 * NOTE: Currently the code in SubgraphPartitioner relies on the global
 * schedule to determine how to lower CallOps. It will look at each instance
 * of a subgraph in the global schedule and look for the positions of the
 * subgraph's input/output copies, etc. This means not all copying strategies
 * are currently valid. In particular:
 *
 * * All copies have to be inserted after the associated enter node and before
 *   the associated exit node.
 * * The placement of copies must be identical for each instance of a subgraph
 *   in the global schedule. To this end, ignore information about the parent
 *   graph of the copy.
 * * Due to an implementation limitation, only subgraphs that belong to
 *   CallOps can be partitioned. For other subgraphs, input copies should
 *   happen before the execution of the subgraph and output copies should
 *   happen after.
 *
 * Another thing to bear in mind is that by moving a copy into the middle of
 * the expansion of a subgraph you are partitioning that subgraph, which may
 * restrict Poplar's ability to overlap IO and compute tasks.
 **/
class SubgraphCopyingStrategy {
public:
  // Default constructor.
  SubgraphCopyingStrategy();
  // Default destructor.
  virtual ~SubgraphCopyingStrategy();

  /**
   * Set the IR dependency to use.
   */
  virtual void setIr(const Ir *);

  /**
   * Set the LivenessAnalyzer dependency to use.
   */
  virtual void setLivenessAnalyzer(const LivenessAnalyzer *);

  /**
   * Do some initial analysis before use, if required.
   */
  virtual void apply();

  /**
   * For a given LivenessNode (which could be entering a subgraph, exiting
   * a subgraph or a normal op) and a set of CopyInput, CopyOutput and
   * CopyModified nodes, determine which copies must precede the node (and
   * in which order). Return a list of indices into pendingCopies.
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertBeforeNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const = 0;

  /**
   * For a given LivenessNode (which could be entering a subgraph, exiting
   * a subgraph or a normal op) and a set of CopyInput, CopyOutput and
   * CopyModified nodes, determine which copies must come after the node (and
   * in which order). Return a list of indices into pendingCopies.
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertAfterNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const = 0;

protected:
  // Ir instance (dependency).
  const Ir *ir;
  // LivenessAnalyzer instance (dependency).
  const LivenessAnalyzer *liveness;
};

/**
 * Implementation of a SubgraphCopyingStrategy that tries to keep subgraphs
 * whole by ensuring all input copies happen before the subgraph and all output
 * copies happen after the subgraph.
 */
class OnEnterAndExitSubgraphCopyingStrategy : public SubgraphCopyingStrategy {
public:
  // Default constructor.
  OnEnterAndExitSubgraphCopyingStrategy();
  // Default destructor.
  virtual ~OnEnterAndExitSubgraphCopyingStrategy();

  /**
   * See SubgraphCopyingStrategy().
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertBeforeNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const override;

  /**
   * See SubgraphCopyingStrategy().
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertAfterNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const override;

private:
  // Helper function.
  void addMatching(const LivenessNode &node,
                   const LivenessAnalyzer::PendingCopies &pendingCopies,
                   OpStatus status,
                   std::vector<size_t> &result) const;
};

/**
 * Implementation of a SubgraphCopyingStrategy that moves input copies as late
 * as possible and output copies as early as possible in an attempt to reduce
 * the amount of live memory.
 */
class JustInTimeSubgraphCopyingStrategy : public SubgraphCopyingStrategy {
public:
  // Default constructor.
  JustInTimeSubgraphCopyingStrategy();
  // Default destructor.
  virtual ~JustInTimeSubgraphCopyingStrategy();

  /**
   * Do some initial analysis before use.
   */
  virtual void apply() override;

  /**
   * See SubgraphCopyingStrategy().
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertBeforeNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const override;

  /**
   * See SubgraphCopyingStrategy().
   */
  virtual std::vector<size_t> getIndicesOfCopiesToInsertAfterNode(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const override;

private:
  // Enum class to help make it clear what getAssociatedCopies's parameters
  // mean from the call site.
  enum class TensorPosition { Consuming = 0, Producing };

  // Enum class to help make it clear what getAssociatedCopies's parameters
  // mean from the call site.
  enum class FilterSetting { Include = 0, Ignore };

  // Deal with (potentially recursively) inserting CopyInputs to satisfy
  // the consumed tensors of a normal op.
  std::vector<size_t> addCopyInputsForNormalOp(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies) const;

  // Deal with (potentially recursively) inserting CopyOutputs
  // for tensors that are no longer modified in a subgraph after a position.
  std::vector<size_t> addCopyOutputsForSchedPosition(
      const LivenessNode &node,
      const LivenessAnalyzer::PendingCopies &pendingCopies,
      const Graph &graph,
      std::vector<Op *>::const_iterator schedPos) const;

  // Get the indices of copies in pendingCopies that 'produce' or 'consume' a
  // tensorId. Ignore indices in ignoreIndices, and ignore types of copies as
  // the filter settings.
  std::vector<size_t>
  getAssociatedCopies(const TensorId &tensorId,
                      TensorPosition tensorPos,
                      const LivenessAnalyzer::PendingCopies &pendingCopies,
                      const std::deque<size_t> &ignoreIndices,
                      const FilterSetting copyInputFilter,
                      const FilterSetting copyOutputFilter,
                      const FilterSetting copyModifiedFilter) const;

  std::set<TensorId> getAliases(const Graph &graph, const TensorId &id) const;

  // Helper function to determine if any of the ops in a range of ops produce
  // a specific tensor.
  bool isProducedInOpRange(std::vector<Op *>::const_iterator begin,
                           std::vector<Op *>::const_iterator end,
                           TensorId outputId) const;

  // Helper function to determine if any of the ops in a range of ops modify a
  // tensor, taking into account aliasing.
  bool isModifiedInOpRange(std::vector<Op *>::const_iterator begin,
                           std::vector<Op *>::const_iterator end,
                           Tensor *tensor) const;

  // Get the tensor 'consumed' by a CopyInput, CopyOutput or CopyModied node.
  TensorId getConsumedTensorForCopy(const LivenessNode &node) const;
  // Get the tensor 'produced' by a CopyInput, CopyOutput or CopyModied node.
  TensorId getProducedTensorForCopy(const LivenessNode &node) const;

  // Helper function to determine if graph is partitionable.
  bool isPartitionable(const Graph &graph) const;

  // TODO T40060: Replace use of chain-based aliasing.
  AliasesMap aliasesMap;

  // Cache from SubgraphPartitioner::isPartitionable.
  std::map<std::string, bool> isPartitionableCache;
};

} // namespace liveness
} // namespace popart

#endif
