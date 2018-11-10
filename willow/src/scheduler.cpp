#include <poponnx/error.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/scheduler_topological.hpp>

namespace willow {

Scheduler::Scheduler() {}

std::unique_ptr<Scheduler> Scheduler::getScheduler(
    const std::string& scheduler) {
  if (scheduler == "default") {
    return std::unique_ptr<Scheduler>(new TopologicalScheduler());
  }
  throw error("Unrecognised scheduler " + scheduler);
}

} // namespace willow
