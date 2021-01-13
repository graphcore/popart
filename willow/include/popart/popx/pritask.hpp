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

// Conjunctive normal form (CNF) of dependencies
class PriTaskDependency {
public:
  PriTaskDependency(TaskId taskId, DependencyType type);
  PriTaskDependency(std::set<TaskId> taskIds, DependencyType type);

  DependencyType getType() const { return type; }

  bool satisfiedBy(TaskId taskId) const {
    return taskIds.find(taskId) != taskIds.end();
  }

  const std::set<TaskId> &getTaskIds() const { return taskIds; }

  bool operator==(PriTaskDependency const &rhs) const;

private:
  DependencyType type;
  // Dependency is fulfilled if any of the tasks are done (CNF)
  std::set<TaskId> taskIds;
};

std::ostream &operator<<(std::ostream &oss, const DependencyType &dt);
std::ostream &operator<<(std::ostream &oss, const PriTaskDependency &ptd);

/** A Pritask is a task which has a priority and a set of dependent tasks. */
class PriTask {
public:
  double priority;
  // the name of this task, must be a unique identifier
  TaskId name;
  // the names of tasks which MUST appear before this task in a linearisation
  std::vector<PriTaskDependency> dependsOn;
  std::function<SequenceMap()> f;

  PriTask() = default;
  PriTask(double priority,
          TaskId,
          const std::vector<PriTaskDependency> &deps,
          const std::function<SequenceMap()> &funcInst);

  // Remove dep from dependsOn.
  // Returns true if a dependency has been removed.
  bool removeDep(const TaskId &dep);

  // Remove all dependencies of type from dependsOn.
  // Returns true if a dependency has been removed.
  bool removeDep(DependencyType type);

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
