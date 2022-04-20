// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_PRITASK_HPP
#define GUARD_PRITASK_HPP

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <map>
#include <queue>
#include <set>
#include <snap/Program.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <popart/taskid.hpp>

namespace snap {
class Graph;
} // namespace snap

namespace popart {

/**
 * This class offers a level of indirection that allows us to grow Poplar
 * sequences in two stages (see comments in irlowering.cpp). This class
 * essentially maps pointers to Poplar sequences to a local sequences. We grow
 * ops in the local data structure and later lower this into the real Poplar
 * sequences.
 *
 * While the vast majority of ops are lowered into one sequence, some ops
 * (notably CallOp) may be lowered into multiple sequences. To facilitate this
 * the interface to grow ops is now an iterator of sequences rather than
 * single sequence. The main complexity of this class is there to be able to
 * provide an iterator to local sequences while also remembering the mapping
 * from Poplar sequences to local sequences.
 *
 * NOTE: Poplar sequences passed to a SequenceMap will be referred to by a
 * pointer and hence this class implicitly assumes all such Poplar sequences
 * are alive for the lifetime of this object. This is typically achieved
 * by passing Poplar sequences that reside in popprograms.cpp.
 */
class SequenceMap {
public:
  // Shorthand.
  using Sequence         = snap::program::Sequence;
  using Sequences        = std::vector<Sequence>;
  using SequenceIterator = typename std::vector<Sequence>::iterator;
  using SequenceInterval =
      typename std::pair<SequenceIterator, SequenceIterator>;

  SequenceMap(snap::Graph &);

  // Add a local sequence mapping for a single Poplar sequence. Note that the
  // Poplar sequence must exist for the whole lifetime of the SequenceMap. Note,
  // sequences that need to be accessed by iterator need to be loaded via
  // addScopeFragments.
  void addSingleSequence(Sequence &sequence);
  // Add multiple local sequences for a vector of Poplar sequences. Also add the
  // ability to provide iterators for the local sequences for any of the
  // sequences. Poplar sequences must exist for the lifetime of this class.
  void addScopeFragments(Sequences &sequences);

  // If there already is a local sequence for this sequence pointer then
  // return a reference to that local sequence. If not, create a local sequence
  // first and return a reference. Note, sequences that need to be accessed by
  // iterator need to be loaded via addScopeFragments.
  Sequence &getSequence(Sequence *seq);

  // Return a begin/end iterator of local sequences for a starting Poplar
  // sequence. This function is useful for lowering spreading the lowering
  // of an op over multiple local Sequences that later have to be
  // added to Poplar sequences. This is currently really used by CallOps.
  // If there already is a local sequence mapping for this Poplar sequence
  // pointer then return the largest interval possible. If not, add a
  // singleton vector containing a local sequence and return it as a unit
  // interval.
  SequenceInterval operator[](Sequence *seq);

  // Build and return a direct mapping from final Poplar sequence to a local
  // sequence. This is only meant to be used for mapping the latter into the
  // former after all ops have been lowered. This could potentially done more
  // neatly by making this class iterable, instead.
  std::map<Sequence *, Sequence *> getFullSequenceMap();

private:
  // A vector of vectors from which we draw sequences.
  std::vector<Sequences> localSeqs;
  // Mapping from pointers to Poplar sequences to a tuple containing 1) the
  // index into localSeqs and 2) the offset into the vector at that index.
  std::map<Sequence *, std::tuple<size_t, size_t>> indexMap;
  // Counter for debug context ids.
  static int debugCtxNo;

  snap::Graph &graph;
};

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

/** A Pritask is a task which has a priority and a set of dependant tasks. */
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

namespace popx {
class IrLowering;
}

class PriTasks {
public:
  std::unordered_map<TaskId, PriTask> tasksMap;
  void add(const PriTask &t);
  bool contains(const TaskId &);

  // return the tasks in an order of descending priority as far
  // as possible, subject to all dependencies being satisfied.
  // the algorithm used is a variant of  Kahn's algorithm.
  // If allowFallback is set, any task with altenativeCreator set can be
  // replaced with dependency free version using the creator. In this case,
  // pritasks in taskMap will be modifed.
  // An error is thrown if not linearizable (schedulable).
  std::vector<PriTask> getLinearised(std::set<DependencyType> dependencies,
                                     popx::IrLowering &irLowering,
                                     bool allowFallback = false);
  PriTasks() = default;

private:
  // Returns strong component elements within unscheduled based on dependencies
  // specified by dependantOf: in the event of a cyclic dependency, any
  // candidate which can be replaced with a dependency free version must
  // be part of a strong component or the replacement will never resolve the
  // cycle.
  static std::vector<TaskId> getStongComponentElements(
      const std::vector<TaskId> &unscheduled,
      const std::unordered_map<TaskId, std::set<TaskId>> &dependantsOf);

  // Perform actual scheduling (topological sorting) of the tasks
  // Allows recursive calling in case of allowFalling being true.
  void
  schedule(std::priority_queue<PriTask> &pq,
           std::vector<PriTask> &linearisedTasks,
           const std::unordered_map<TaskId, std::set<TaskId>> &dependantsOf,
           std::unordered_map<TaskId, PriTask> &taskMapsCopy,
           popx::IrLowering &irLowering,
           bool allowFallback);
};

} // namespace popart

#endif
