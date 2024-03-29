// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SCHEDULER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SCHEDULER_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace poprithms {
namespace schedule {
namespace shift {
class ScheduleCache;
}
} // namespace schedule
} // namespace poprithms

namespace popart {
class Graph;
class GraphId;
class Op;

enum class RequireOptimalSchedule; /*
  Yes = true,
  No = false
*/

class Scheduler {

public:
  Scheduler();
  ~Scheduler();

  /**
   * Get a valid schedule based on the settings.
   * \param opsBeforeKey topological constraints external to \a graph
   *
   * \param  graph the Graph whose Ops are to be scheduled
   *
   * \param  requireOptimalSchedule whether the true optimal schedule is
   *         required, which is expensive to compute, or merely any valid
   *         topological ordering
   *
   * \param  respectExecutionPhase if true: Ops must appear in ascending order
   *of ping-pong phase
   *
   * \param  kahnTieBreaker the initial scheduling is done with Kahn's
   *algorithm. When several Ops are free to be scheduled, this controls which
   *one is chosen
   *
   * \param  timeLimitSeconds the maximum permitted time for schedule
   *         improvement, before it must be returned
   *
   * \param  swapLimitCount the maximum number of schedule-improving swaps
   *         allowed before a schedule must be returned
   *
   * \return vector of operations representing the schedule
   *
   **/

  std::vector<Op *>
  getSchedule(const OpsBeforeKey &opsBeforeKey,
              const Graph &graph,
              const RequireOptimalSchedule requireOptimalSchedule,
              bool respectExecutionPhase,
              double timeLimitSeconds,
              int64_t swapLimitCount,
              const std::string &kahnTieBreaker);

  /**
   * Get a valid schedule based on the settings, and make it final.
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
   * \param kahnTieBreaker the initial scheduling is done with Kahn's algorithm.
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
  void finalizeSchedule(const OpsBeforeKey &opsBeforeKey,
                        const Graph &graph,
                        const RequireOptimalSchedule requireOptimalSchedule,
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

  bool isFinalized() const { return finalized; }

private:
  std::unique_ptr<poprithms::schedule::shift::ScheduleCache> cacher;
  std::map<GraphId, std::vector<Op *>> finalizedSchedules;
  bool finalized;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SCHEDULER_HPP_
