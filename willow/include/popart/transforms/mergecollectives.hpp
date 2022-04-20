// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MERGECOLLECTIVESTRANSFORM_HPP
#define GUARD_NEURALNET_MERGECOLLECTIVESTRANSFORM_HPP
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>
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
  virtual bool apply(Graph &graph) const override;
  std::vector<Op *> applyToOps(Graph &graph,
                               const std::set<OpId> includeOps) const;
  virtual std::size_t getId() const override { return id(); }
  virtual std::string getName() const override {
    return "MergeCollectivesTransform";
  }

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
   * \param modifiesIndexInplace specifies which indices are used inplace
   *        by the original ops
   * \param outInfoFromBaseOps is the output information for each output tensor
   *        collected from the ops with which base op will be merged.
   * \param inputVirtualGraphIdAndTileSet the input virtual graph and tile set
   *        information collected from the ops that will be merged
   * \param outputVirtualGraphIdAndTileSet the output virtual graph and tile set
            information collected from the ops that will be merged
   * \return a unique pointer to the new multi-collective op
   */
  template <typename MultiOpType, typename BaseType>
  std::unique_ptr<MultiOpType> constructMultiOp(
      BaseType *baseOp,
      std::vector<bool> modifiesIndexInplace,
      std::vector<TensorInfo> outInfoFromBaseOps,
      std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet,
      std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet) const;
};
} // namespace popart

#endif
