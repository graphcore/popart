#ifndef GUARD_NEURALNET_SCHEDULER_HPP
#define GUARD_NEURALNET_SCHEDULER_HPP

#include <vector>
#include <poponnx/names.hpp>

namespace poponnx {

class Scheduler {

public:
  Scheduler(const Ir *pir_) : pir(pir_) {}

  // get as much of a schedule as possible. If the Ops with all their
  // ordering constraints form a DAG, this schedule will contain all
  // the Ops
  std::vector<Op *> getPartialOpSchedule(const OpsBeforeKey &) const;

private:
  const Ir *pir;
};

} // namespace poponnx

#endif
