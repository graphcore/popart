// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCHEDULER_HPP
#define GUARD_NEURALNET_SCHEDULER_HPP

#include <memory>
#include <vector>
#include <popart/names.hpp>
#include <popart/scheduler_requireoptimal.hpp>

namespace poprithms {
namespace schedule {
namespace shift {
class SolutionCache;
}
} // namespace schedule
} // namespace poprithms

namespace popart {

enum class RequireOptimalSchedule; /*
  Yes = true,
  No = false
*/

class Scheduler {

public:
  Scheduler();
  ~Scheduler();

  /**
   * \param opsBeforeKey topological constraints external to \a graph
   *
   * \param  graph the Graph whose Ops are to be scheduled
   *
   * \param requireOptimalSchedule whether the true optimal schedule is
   *        required, which is expensive to compute, or merely any valid
   *        topological ordering
   *
   * \param respectExecutionPhase if true: Ops must appear in ascending order of
   *        ping-pong phase
   *
   * \param khanTieBreaker the initial scheduling is done with Kahn's algorithm.
   *        When several Ops are free to be scheduled, this controls which one
   *        is chosen
   *
   * \param timeLimitSeconds the maximum permitted time for schedule
   *        improvement, before it must be returned
   *
   * \param swapLimitCount the maximum number of schedule-improving swaps
   *        allowed before a schedule must be returned
   *
   **/

  std::vector<Op *> getSchedule(const OpsBeforeKey &opsBeforeKey,
                                const Graph &graph,
                                bool respectExecutionPhase,
                                double timeLimitSeconds,
                                int64_t swapLimitCount,
                                const std::string &kahnTieBreaker);
  /**
   *  to determine if a Graph is schedulable - that is, it contains no cycles -
   *  is a simpler problem than finding the schedule in the first place. This
   *  function should be used to determine if a Graph is schedulable as it
   *  executes significantly faster than getSchedule
   *  */
  bool isSchedulable(const OpsBeforeKey &,
                     const Graph &,
                     bool respectExecutionPhase) const;

private:
  std::unique_ptr<poprithms::schedule::shift::SolutionCache> cacher;
};

} // namespace popart

#endif
