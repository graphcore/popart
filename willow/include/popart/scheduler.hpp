#ifndef GUARD_NEURALNET_SCHEDULER_HPP
#define GUARD_NEURALNET_SCHEDULER_HPP

#include <vector>
#include <popart/names.hpp>

namespace popart {

class Scheduler {

public:
  Scheduler() {}

  // get as much of a schedule as possible. If the Ops with all their
  // ordering constraints form a DAG, this schedule will contain all
  // the Ops
  // respectPriorities: Graph will be scheduled optimally if enabled, including
  //                    topological constraints. Otherwise, any possible
  //                    schedule will be returned.
  // respectPingPongPhase: Schedule with absolute order on the ping pong phases.
  std::vector<Op *> getPartialOpSchedule(const OpsBeforeKey &,
                                         const Graph &,
                                         bool respectPriorities,
                                         bool respectPingPongPhase) const;
};

} // namespace popart

#endif
