#ifndef GUARD_SCHEDULER_HPP
#define GUARD_SCHEDULER_HPP

#include <memory>

#include <poponnx/names.hpp>

namespace willow {

class Op;
class Tensors;

using OpMap = std::map<OpId, std::unique_ptr<Op>>;

/**
 * An interface for a scheduler.  It takes a set of operations and constructs
 * a schedule (ordered list) of them, based on their dependencies and other
 * information as required.
 */
class Scheduler {
public:
  Scheduler();

  virtual ~Scheduler() {}

  virtual std::vector<Op *> getSchedule(const OpMap &ops,
                                        const Tensors &tensors) const = 0;

  static std::unique_ptr<Scheduler> getScheduler(const std::string &scheduler);
};

} // namespace willow

#endif
