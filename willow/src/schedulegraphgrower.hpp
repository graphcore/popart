// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCHEDULEGRAPHGROWER_HPP
#define GUARD_NEURALNET_SCHEDULEGRAPHGROWER_HPP

#include <unordered_map>
#include <poprithms/schedule/shift/scheduledgraph.hpp>
#include <popart/names.hpp>

namespace popart {

class Graph;

/**
 * A class for creating a poprithms::schedule::shift::Graph, and the handles
 * between it an the PopART Graph from which it is derived.
 * */
class ShiftGraphGrower {

private:
  const popart::Graph &pg;
  const uint64_t nOps;
  const std::vector<TensorId> allPopartTensorIds;

  // Populate, 1-1 for popart::Op <-> poprithms Op and
  //               popart::Tensor <-> poprithms Alloc.
  std::unordered_map<popart::Tensor *, poprithms::schedule::shift::AllocAddress>
      allocAddresses;
  std::unordered_map<popart::Op *, poprithms::schedule::shift::OpAddress>
      opAddresses;
  std::vector<Op *> addressToOp;
  poprithms::schedule::shift::Graph g;
  poprithms::schedule::shift::ScheduledGraph scheduledShiftGraph;

public:
  ShiftGraphGrower(const Graph &_pg_);

  // Get the schedule from the shift::Graph as a vector of Op pointers.
  // The shift::Graph must have already been initialised through a call to
  // `ShiftGraphGrower::initialize`.
  std::vector<Op *> getSchedule() const;

  void initialize(const poprithms::schedule::shift::Settings &settings,
                  poprithms::schedule::shift::SolutionCache &cache);

  bool isSchedulable() const;

  std::string getSerializationString() const;

  Op *toOp(poprithms::schedule::shift::OpAddress a) const;

  void setBasic();

  void annotateExecutionPhase();

  void annotateExecutionContext();

  void annotatePipelineStages();

  void annotateAccumulateOuterFragmentOps();

  // The general setting of an op's scheduledPreLoss setting may look like:
  //
  //         scheduledPreLoss?
  // Op0     Yes
  // Op1     Yes
  //     ...
  // Loss    No
  // Loss'   No
  //     ...
  // OpN-1   No
  // OpN     No
  //
  // However, the loss final loss can be computed arbitrarily, and therefore
  // gradient operations can be grown in the auto-diff transform that do not
  // have a dependency of any operations with a path to the loss. For example,
  // if:
  //   loss = Mul(ReduceSum(Reshape(probs)), const)
  // the ReshapeGrad, ReduceSumGrad and MulGrad operations that produce the
  // gradient of 'loss' tensor do not depend on operations with a path to the
  // 'loss' tensor. Therefore they can be scheduled early, leading to corrupted
  // scheduledPreLoss settings, such as:
  //
  //         scheduledPreLoss?
  // Op0     Yes
  // Loss'   No
  // Op1     No
  //     ...
  // Loss    No
  //     ...
  // OpN-1   No
  // OpN     No.
  //
  // The implicit recomputation transform depends on this setting
  // correctly indicating whether an op is in the forward or backward
  // pass, so insert scheduler constraints to prevent this from happening.
  void annotateToLossFromLoss();

  void annotatePriorities();

  void appendGCons(const OpsBeforeKey &gCons);

  poprithms::schedule::shift::Graph getGraph() const { return g; }
  const poprithms::schedule::shift::Graph &getGraphRef() const { return g; }
};

} // namespace popart

#endif
