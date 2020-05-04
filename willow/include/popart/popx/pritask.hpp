// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_PRITASK_HPP
#define GUARD_PRITASK_HPP

#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <poplar/Program.hpp>
#include <popart/names.hpp>

namespace popart {

using SequenceMap =
    std::map<poplar::program::Sequence *, poplar::program::Sequence>;

enum class DependencyType {
  // Depends on output of other Op
  Output = 0,
  // Depends on tensor initialisation (weight)
  Tensor,
  // Depends on scheduler choice
  Scheduler,
  // Depends on subgraph ops
  SubGraph
};

/** A Pritask is a task which has a priority and a set of dependent tasks. */
class PriTask {
public:
  double priority;
  // the name of this task, must be a unique identifier
  TaskId name;
  // the names of tasks which MUST appear before this task in a linearisation
  std::vector<std::pair<TaskId, DependencyType>> dependsOn;
  std::function<SequenceMap()> f;

  PriTask() = default;
  PriTask(double priority,
          TaskId,
          const std::vector<std::pair<TaskId, DependencyType>> &deps,
          const std::function<SequenceMap()> &funcInst);

  // remove dep from dependsOn.
  void removeDep(const TaskId &dep);
  void removeDep(DependencyType type);
  std::set<TaskId> getDependenciesOfTypes(std::set<DependencyType>);
};

// returns true if "a" has lower priority than "b"
bool operator<(const PriTask &a, const PriTask &b);

class PriTasks {
public:
  std::unordered_map<TaskId, PriTask> tasksMap;
  void add(const PriTask &t);
  bool contains(const TaskId &);

  // return the tasks in an order of descending priority as far
  // as possible, subject to all dependencies being satisfied.
  // the algorithm used is a variant of  Kahn's algorithm.
  // An error is thrown if not linearizable (schedulable).
  std::vector<PriTask>
  getLinearised(std::set<DependencyType> dependencies) const;
  PriTasks() = default;
};

} // namespace popart

#endif
