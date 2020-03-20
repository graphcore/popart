// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCHEDULER_HPP
#define GUARD_NEURALNET_SCHEDULER_HPP

#include <vector>
#include <popart/names.hpp>

namespace popart {

class ScheduleCacher;

class Scheduler {

public:
  Scheduler();
  ~Scheduler();

  // - opsBeforeKey
  // topological constraints external to "graph"
  //
  // - graph
  // the Graph whose Ops are to be scheduled
  //
  // - respectPingPongPhase
  // if true: Ops must appear in ascending order of ping-pong phase
  //
  // - timeLimitSeconds
  // the maximum permitted time for schedule improvement, before it must be
  // returned
  //
  // - swapLimitCount
  // the maximum number of schedule-improving swaps allowed before a schedule
  // must be returned
  //
  // - khanTieBreaker
  // the initial scheduling is done with Kahn's algorithm. When several Ops are
  // free to be scheduled, this controls which one is chosen
  //
  std::vector<Op *> getSchedule(const OpsBeforeKey &opsBeforeKey,
                                const Graph &graph,
                                bool respectPingPongPhase,
                                double timeLimitSeconds,
                                int64_t swapLimitCount,
                                const std::string &kahnTieBreaker);

  // to determine if a Graph is schedulable - that is, it contains no cycles -
  // is a simpler problem than finding the schedule in the first place. This
  // function should be used to determine if a Graph is schedulable as it
  // executes significantly faster than getSchedule
  bool isSchedulable(const OpsBeforeKey &,
                     const Graph &,
                     bool respectPingPongPhase) const;

private:
  std::unique_ptr<ScheduleCacher> cacher;
};

} // namespace popart

#endif
