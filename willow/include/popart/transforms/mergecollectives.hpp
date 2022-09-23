// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGECOLLECTIVES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGECOLLECTIVES_HPP_
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/multi_replicatedallgather.hpp>
#include <popart/op/collectives/multi_replicatedallreduce.hpp>
#include <popart/op/collectives/multi_replicatedreducescatter.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/operators.hpp>
#include <popart/scheduler.hpp>
#include <popart/topocons.hpp>
#include <popart/transforms/transform.hpp>

#include "popart/names.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class Graph;
class Op;
class TensorInfo;

/**
 * A transform for merging multiple compatible collective operations
 * into a single larger collective operation.
 * Ops are only merged if they apear in contiguous order in the schedule.
 *
 * Currently supported collective types:
 *  - ReplicatedAllReduceOp
 */
class MergeCollectivesTransform : public Transform {
public:
  static std::size_t id();

  MergeCollectivesTransform() : Transform() {}
  ~MergeCollectivesTransform() override {}
  bool apply(Graph &graph) const override;
  std::vector<Op *> applyToOps(Graph &graph,
                               const std::set<OpId> includeOps) const;
  std::size_t getId() const override { return id(); }
  std::string getName() const override { return "MergeCollectivesTransform"; }

  /**
   * Confirm that two collective ops of the same BaseType use the same
   * collective operation i.e. ADD, MUL etc. If the BaseType does not require a
   * collective op (gather), return true \param A the first op \param B the
   * second op \return true is A and B use the same collective operation or both
   * use none
   */
  template <typename BaseType>
  bool collectiveOpCheck(BaseType *A, BaseType *B) const;

  /**
   * Given a collective operation, attempt to merge it with other compatible
   * collective ops which are tied (in the schedule) to the current op.
   * \param baseOp a collective op that should be merged
   * \param opSchedule the schedule of all (collective) ops in the graph
   * \return pointer the constructed op
   */
  template <typename MultiOpType, typename BaseType>
  Op *attemptToMergeOnOp(BaseType *baseOp,
                         std::vector<Op *>::iterator &schedulePos,
                         std::vector<Op *> &opSchedule) const;

  /**
   * Constructs a new MultiOpType which will replace the baseOp and
   *  all matching ops
   * \param baseOp is the operation to be replaced
   * \param outInfoFromBaseOps is the output information for each output tensor
   *        collected from the ops with which base op will be merged.
   * \param inputVirtualGraphIdAndTileSet the input virtual graph and tile set
   *        information collected from the ops that will be merged
   * \param outputVirtualGraphIdAndTileSet the output virtual graph and tile set
            information collected from the ops that will be merged
   * \param matchingOps the vector of matching ops
   * \return a unique pointer to the new multi-collective op
   */
  template <typename MultiOpType, typename BaseType>
  std::unique_ptr<MultiOpType> constructMultiOp(
      BaseType *baseOp,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet,
      std::vector<BaseType *> matchingOps) const;
};
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_MERGECOLLECTIVES_HPP_
