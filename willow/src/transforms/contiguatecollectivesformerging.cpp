// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include <popart/transforms/contiguatecollectivesformerging.hpp>

#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/graphutils.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/op/collectives/replicatedallgather.hpp"
#include "popart/op/collectives/replicatedallreduce.hpp"
#include "popart/op/collectives/replicatedreducescatter.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/topocons.hpp" // IWYU pragma: keep
#include "popart/transforms/transform.hpp"
#include "popart/util.hpp"

namespace popart {
std::size_t ContiguateCollectivesTransform::id() {
  return typeid(ContiguateCollectivesTransform).hash_code();
}

bool ContiguateCollectivesTransform::apply(Graph &graph) const {
  std::set<OpId> includeOps;
  for (OpId id : graph.getOpIds()) {
    includeOps.insert(id);
  }
  applyToOps(graph, includeOps);
  return true;
}

std::vector<Op *> ContiguateCollectivesTransform::applyToOps(
    Graph &graph,
    const std::set<OpId> includeOps) const {

  auto &opts = graph.getIr().getSessionOptions();
  if (opts.accumulateOuterFragmentSettings.schedule ==
      AccumulateOuterFragmentSchedule::Serial) {
    throw error(
        "Incompatible accumulateOuterFragmentSchedule used with session "
        "option: options.replicatedCollectivesSettings."
        "prepareScheduleForMergingCollectives. "
        "Specifically SessionOptions::accumulateOuterFragmentSettings.schedule "
        "can not be set to AccumulateOuterFragmentSchedule::Serial");
  }

  std::set<Op *, POpCmp> opsToProcess;
  std::vector<Op *> schedule =
      graph.getOpSchedule({}, RequireOptimalSchedule::Yes);
  for (auto op : schedule) {
    if (op->isConvertibleTo<CollectivesBaseOp>() &&
        includeOps.count(op->id) > 0) {
      opsToProcess.insert(op);
    }
  }

  // Iterate through the ops, removing those that have already been made
  // contiguous
  while (!opsToProcess.empty()) {
    Op *op = *opsToProcess.begin();
    if (auto opC = dynamic_cast<ReplicatedAllReduceOp *>(op)) {
      processOp(opC, schedule, opsToProcess);
    } else if (auto opC = dynamic_cast<ReplicatedAllGatherOp *>(op)) {
      processOp(opC, schedule, opsToProcess);
    } else if (auto opC = dynamic_cast<ReplicatedReduceScatterOp *>(op)) {
      processOp(opC, schedule, opsToProcess);
    }
    opsToProcess.erase(op);
  }
  // Does not create any new ops
  return {};
}

template <typename BaseType>
bool ContiguateCollectivesTransform::checkCollectiveOp(BaseType *baseOp,
                                                       BaseType *candidate) {
  return baseOp->getCollectiveOp() == candidate->getCollectiveOp();
}

template <>
bool ContiguateCollectivesTransform::checkCollectiveOp<ReplicatedAllGatherOp>(
    ReplicatedAllGatherOp *baseOp,
    ReplicatedAllGatherOp *candidate) {
  // Gather does not use a collective op
  return true;
}

template <typename BaseType>
std::set<BaseType *, POpCmp> ContiguateCollectivesTransform::lookForMatchingOps(
    BaseType *baseOp,
    const std::vector<Op *> &schedule,
    std::set<Op *, POpCmp> &opsToProcess) {

  auto &graph = baseOp->getGraph();
  std::set<BaseType *, POpCmp> allMatches{baseOp};
  std::vector<Op *> allDataDependencies{graph.getOp(baseOp->id)};
  for (Op *op : schedule) {
    if (opsToProcess.count(op) > 0) {
      auto candidate = dynamic_cast<BaseType *>(op);
      if (candidate && candidate->id != baseOp->id) {
        // There should be no data inconsistencies introduced by the merge
        bool dataDependencyCheck =
            !graphutils::hasDataDependency(op, schedule, allDataDependencies);

        // Data types must match
        auto dtype = baseOp->inTensor(baseOp->getInIndex())->info.data_type();
        bool dtypeCheck =
            dtype == op->inTensor(candidate->getInIndex())->info.data_type();

        // Both must have the same collective operation type on the same group
        bool groupCheck =
            candidate->getReplicaGrouping() == baseOp->getReplicaGrouping();
        bool collTypeCheck             = checkCollectiveOp(baseOp, candidate);
        auto &requiredExecutionContext = baseOp->settings.executionContext;
        bool executionContextCheck =
            candidate->settings.executionContext == requiredExecutionContext;

        if (groupCheck && collTypeCheck && dtypeCheck && dataDependencyCheck &&
            executionContextCheck) {
          allMatches.insert(candidate);
          allDataDependencies.emplace_back(op);
        }
      }
    }
  }
  return allMatches;
}

template <typename BaseType>
void ContiguateCollectivesTransform::processOp(
    BaseType *baseOp,
    const std::vector<Op *> &schedule,
    std::set<Op *, POpCmp> &opsToProcess) const {
  auto &graph = baseOp->getGraph();

  // Find matching ops and place them in a sorted set
  // this ensures that the the matches are processed in a
  // deterministic order
  std::set<BaseType *, POpCmp> allMatches =
      lookForMatchingOps(baseOp, schedule, opsToProcess);

  std::vector<std::string> allMatchNames;
  for (auto op : allMatches) {
    allMatchNames.emplace_back("\t" + op->debugName() + "\n");
  }
  logging::info(
      "[ContiguateCollectivesForMerging] Tying op {} with matches: \n{}",
      baseOp->debugName(),
      allMatchNames);

  // Force the ops to follow one another in the schedule
  BaseType *before = *allMatches.begin();
  for (BaseType *after : allMatches) {
    // baseOp is in allMatches, so need to prevent creation of A before A type
    // constraint
    if (after->id != before->id) {
      graph.topoCons->insert(before, after, true);
    }
    before = after;
    opsToProcess.erase(after);
  }
}

namespace {
bool init = Transform::registerTransform(new ContiguateCollectivesTransform);
}

} // namespace popart
