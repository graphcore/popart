// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONTIGUATECOLLECTIVESTRANSFORM_HPP
#define GUARD_NEURALNET_CONTIGUATECOLLECTIVESTRANSFORM_HPP
#include <cstddef>
#include <set>
#include <string>
#include <vector>
#include <popart/pointercomparators.hpp>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;

/**
 * A transform that inserts topological constraints into the graph.
 * These force collective operations which can potentially be merged
 * to be scheduled contiguously (one right after the other) in the schedule.
 *
 * Currently supported collective types:
 *  - ReplicatedAllReduceOp
 *  - ReplicatedReduceScatterOp
 *  - ReplicatedAllGatherOp
 */

class ContiguateCollectivesTransform : public Transform {
public:
  static std::size_t id();

  ContiguateCollectivesTransform() : Transform() {}
  ~ContiguateCollectivesTransform() override {}
  virtual bool apply(Graph &graph) const override;
  std::vector<Op *> applyToOps(Graph &graph,
                               const std::set<OpId> includeOps) const;
  virtual std::size_t getId() const override { return id(); }
  virtual std::string getName() const override {
    return "ContiguateCollectivesTransform";
  }

  /**
   * Check whether two ops use the same collective operator
   * \param baseOp against which to compare the candidate op
   * \param candidate op of the same type as baseOp
   * \return true, if the two ops use the same collective operator
   *         or if neither uses a collective operator
   */
  template <typename BaseType>
  static bool checkCollectiveOp(BaseType *baseOp, BaseType *candidate);

  /**
   * Loop through the ops in the schedule and find those matching baseOp
   * to avoid merging the same op twice, make sure it is still in opsToProcess
   * \param baseOp the op that should be merged with other collectives
   * \param schedule the schedule of the (Collective) ops in the graph
   * \param opsToProcess the (Collective) ops that can still be considered for
   * merging
   * \return a vector of collective ops that can be merged with the baseOp
   */
  template <typename BaseType>
  static std::set<BaseType *, POpCmp>
  lookForMatchingOps(BaseType *baseOp,
                     const std::vector<Op *> &schedule,
                     std::set<Op *, POpCmp> &opsToProcess);

  /**
   *  Processing baseOp involves finding all other collective ops in the graph
   *  with which baseOp can be merged, the inserting constraints between the
   *  matching ops and baseOp, that ensure the ops are scheduled contiguously
   *  one after another
   *  \param baseOp is the Op that should be merged with other collectives
   *  \param schedule is a vector of ops sorted in schedule order
   *  \param opsToProcess is set of all other collective ops in the
   *         graph (which are candidates for merging with base op)
   * \return void, modifies the graph of baseOp
   */
  template <typename BaseType>
  void processOp(BaseType *baseOp,
                 const std::vector<Op *> &schedule,
                 std::set<Op *, POpCmp> &opsToProcess) const;
};
} // namespace popart

#endif
