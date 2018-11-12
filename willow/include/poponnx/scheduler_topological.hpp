#ifndef GUARD_TOPOLOGICAL_SCHEDULER_HPP
#define GUARD_TOPOLOGICAL_SCHEDULER_HPP

#include <poponnx/names.hpp>
#include <poponnx/scheduler.hpp>

namespace willow {

/**
 * Essentially Kahn's alogorithm (1962, 56 years ago!),
 * see https://en.wikipedia.org/wiki/Topological_sorting
 * but not quite Kahn's algorithm as it there are some
 * additional constraints on the order of Ops imposed
 * externally. Also not quite Kahn, as the vertices which
 * are ready to be inserted have an insertion "priority"
 * set externally
 */
class TopologicalScheduler : public Scheduler {
public:
  TopologicalScheduler() : Scheduler() {}

  ~TopologicalScheduler() override {}

  std::vector<Op *> getSchedule(const OpMap &ops,
                                const Tensors &tensors) const override;
};

} // namespace willow

#endif
