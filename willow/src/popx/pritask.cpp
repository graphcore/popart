// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <sstream>
#include <poprithms/schedule/scc/scc.hpp>
#include <popart/error.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/pritask.hpp>

namespace popart {

int SequenceMap::debugCtxNo = 0;

SequenceMap::SequenceMap(snap::Graph &g) : graph(g) {}

void SequenceMap::addSingleSequence(Sequence &sequence) {
  // Defensively check for misuse of this class.
  auto it = indexMap.find(&sequence);
  if (it != indexMap.end()) {
    throw error("[SequenceMap] Sequence already mapped");
  }

  // NOTE: It would be nice to use the debug context of the input sequences
  // here but this information is not accessible.
  std::stringstream dbgCtx;
  dbgCtx << "sequence_map/" << debugCtxNo++;
  localSeqs.push_back({Sequence(dbgCtx.str(), graph)});
  indexMap[&sequence] = std::make_tuple(localSeqs.size() - 1, 0);
}

void SequenceMap::addScopeFragments(Sequences &sequences) {
  // Defensively check for misuse of this class.
  for (auto &seq : sequences) {
    auto it = indexMap.find(&seq);
    if (it != indexMap.end()) {
      throw error("[SequenceMap] Sequence in vector already mapped");
    }
  }

  // Add a vector of sequences.
  Sequences seqs;
  for (size_t part = 0; part < sequences.size(); ++part) {
    // NOTE: It would be nice to use the debug context of the input sequences
    // here but this information is not accessible.
    std::stringstream dbgCtx;
    dbgCtx << "sequence_map/" << part << "/" << debugCtxNo;
    seqs.push_back(Sequence(dbgCtx.str(), graph));
  }

  debugCtxNo++;

  localSeqs.push_back(seqs);

  // Now for every sequence pointer we were given, set indexMap.
  for (size_t offset = 0; offset < sequences.size(); ++offset) {
    indexMap[&sequences[offset]] =
        std::make_tuple(localSeqs.size() - 1, offset);
  }
}

SequenceMap::SequenceInterval SequenceMap::operator[](Sequence *seq) {

  auto it = indexMap.find(seq);

  if (it == indexMap.end()) {
    addSingleSequence(*seq);
    it = indexMap.find(seq);
  }

  auto indexIntoLocalSeqs = std::get<0>(it->second);
  auto offset             = std::get<1>(it->second);
  auto &seqs              = localSeqs.at(indexIntoLocalSeqs);

  if (offset >= seqs.size()) {
    // This shouldn't happen.
    throw error("[SequenceMap] Internal mapping error");
  } else {
    // Returning an interval of Sequences in which ops can be grown.
    return SequenceInterval(seqs.begin() + offset, seqs.end());
  }
}

SequenceMap::Sequence &SequenceMap::getSequence(Sequence *seq) {
  return *operator[](seq).first;
}

std::map<SequenceMap::Sequence *, SequenceMap::Sequence *>
SequenceMap::getFullSequenceMap() {
  std::map<Sequence *, Sequence *> result;
  for (auto entry : indexMap) {
    auto indexIntoLocalSeqs = std::get<0>(entry.second);
    auto offset             = std::get<1>(entry.second);
    result[entry.first]     = &localSeqs.at(indexIntoLocalSeqs).at(offset);
  }
  return result;
}

PriTaskDependency::PriTaskDependency(TaskId taskId_, DependencyType type_)
    : type(type_), taskIds({taskId_}) {}
PriTaskDependency::PriTaskDependency(std::set<TaskId> taskIds_,
                                     DependencyType type_)
    : type(type_), taskIds(taskIds_) {}

bool PriTaskDependency::operator==(PriTaskDependency const &rhs) const {
  return type == rhs.type && taskIds == rhs.taskIds;
}

PriTask::PriTask(double p,
                 TaskId n,
                 const std::vector<PriTaskDependency> &d,
                 const std::function<SequenceMap()> &f_)
    : priority(p), name(n), dependsOn(d), f(f_) {}

// Remove all dependencies of TaskId dep
bool PriTask::removeDep(const TaskId &dep) {
  std::vector<PriTaskDependency> newDependsOn;
  newDependsOn.reserve(dependsOn.size());
  for (auto &x : dependsOn) {
    if (!x.satisfiedBy(dep)) {
      newDependsOn.push_back(x);
    }
  }
  bool changed = newDependsOn.size() != dependsOn.size();
  dependsOn    = newDependsOn;
  return changed;
}

bool PriTask::removeDep(DependencyType type) {
  std::vector<PriTaskDependency> newDependsOn;
  newDependsOn.reserve(dependsOn.size());
  for (auto &x : dependsOn) {
    if (x.getType() != type) {
      newDependsOn.push_back(x);
    }
  }
  bool changed = newDependsOn.size() != dependsOn.size();
  dependsOn    = newDependsOn;
  return changed;
}

std::set<TaskId>
PriTask::getDependenciesOfTypes(std::set<DependencyType> depTypes) {
  std::set<TaskId> taskIds;
  for (auto &dep : dependsOn) {
    if (depTypes.find(dep.getType()) != depTypes.end()) {
      auto &depTaskIds = dep.getTaskIds();
      taskIds.insert(depTaskIds.begin(), depTaskIds.end());
    }
  }
  return taskIds;
}

bool operator<(const PriTask &a, const PriTask &b) {
  return (a.priority < b.priority); // we want highest priority off first
}

void PriTasks::add(const PriTask &t) {
  if (tasksMap.find(t.name) != tasksMap.end()) {
    throw error("Already encountered name {} in tasks", t.name);
  }

  std::ostringstream oss;
  oss << "Adding Pritask " << t.name << " <- [ ";
  for (auto dep : t.dependsOn) {
    oss << dep;
  }
  oss << "]";
  logging::devicex::debug(oss.str());

  for (auto x : t.dependsOn) {
    bool allConflicting = x.getTaskIds().size() > 0;
    auto taskIds        = x.getTaskIds();
    for (auto taskId : taskIds) {
      auto found = tasksMap.find(taskId);
      if (found != tasksMap.end()) {
        if (std::find(found->second.dependsOn.begin(),
                      found->second.dependsOn.end(),
                      PriTaskDependency(t.name, DependencyType::Output)) ==
            found->second.dependsOn.end()) {
          allConflicting = false;
        }
      } else {
        allConflicting = false;
      }
    }
    if (allConflicting) {
      throw error("circular PriTask dependency {} <-> {}",
                  logging::join(taskIds.begin(), taskIds.end(), ", "),
                  t.name);
    }
  }
  tasksMap[t.name] = t;
}

bool PriTasks::contains(const TaskId &taskId) {
  return tasksMap.find(taskId) != tasksMap.end();
}

std::vector<TaskId> PriTasks::getStongComponentElements(
    const std::vector<TaskId> &unscheduled,
    const std::unordered_map<TaskId, std::set<TaskId>> &dependantsOf) {

  // For the purpose here, it does not matter if the edges are forward or not
  poprithms::schedule::scc::FwdEdges edges;
  edges.reserve(unscheduled.size());

  std::unordered_map<TaskId, uint64_t> taskToIdx;

  for (size_t idx = 0; idx < unscheduled.size(); idx++) {
    taskToIdx[unscheduled[idx]] = idx;
  }

  for (size_t idx = 0; idx < unscheduled.size(); idx++) {
    edges.emplace_back();

    // BN Because of the use of CNF, this results in more dependencies than
    // there could be, i.e. that "OR" as expressed as if an "AND". A more
    // complex alternative could be to pick the first possibility or a random
    // one
    for (auto &dependant : dependantsOf.at(unscheduled[idx])) {
      if (taskToIdx.count(dependant) == 0) {
        continue;
      }

      edges[idx].push_back(taskToIdx.at(dependant));
    }
  }

  auto strongComponents(
      poprithms::schedule::scc::getStronglyConnectedComponents(edges));

  std::vector<TaskId> strongElements;

  for (auto &component : strongComponents) {
    for (uint64_t element : component) {
      strongElements.push_back(unscheduled[element]);
    }
  }

  return strongElements;
}

void PriTasks::schedule(
    std::priority_queue<PriTask> &pq,
    std::vector<PriTask> &linearisedTasks,
    const std::unordered_map<TaskId, std::set<TaskId>> &dependantsOf,
    std::unordered_map<TaskId, PriTask> &tasksMapCopy,
    popx::IrLowering &irLowering,
    bool allowFallback) {
  // while there are dependency-free tasks which have not been added
  while (!pq.empty()) {
    // 1) add the lowest (highest) priority dep-free task
    linearisedTasks.push_back(tasksMap.at(pq.top().name));
    auto &parent = linearisedTasks.back().name;
    pq.pop();
    // update the dependencies of child tasks, pushing child tasks
    // onto the priority queue if they become dep-free.
    for (auto &child : dependantsOf.at(parent)) {
      bool changed = tasksMapCopy[child].removeDep(parent);
      if (changed && tasksMapCopy[child].dependsOn.size() == 0) {
        pq.push(tasksMapCopy[child]);
      }
    }
  }

  // confirm that the linearisedTasks contains all the tasks.
  // circular dependencies will prevent this.
  if (linearisedTasks.size() == dependantsOf.size()) {
    return;
  }

  std::vector<TaskId> unscheduled;
  unscheduled.reserve(dependantsOf.size() - linearisedTasks.size());

  for (auto &task : tasksMap) {
    bool present = false;

    auto &taskId = task.first;
    for (auto &t : linearisedTasks) {
      if (taskId == t.name) {
        present = true;
        break;
      }
    }

    if (!present) {
      unscheduled.push_back(taskId);
    }
  }

  // If allowFallback, try to replace a task (currently only an initTensorTask)
  // with a dependency-free version and resume the algorithm.
  if (allowFallback) {
    auto unscheduledCycleElements =
        getStongComponentElements(unscheduled, dependantsOf);

    logging::devicex::info("[PriTasks::schedule] Cycle detected, "
                           "trying to remove by using dependency-free "
                           "alternatives: {} unscheduled of which {} "
                           "are in cycles ({})",
                           unscheduled.size(),
                           unscheduledCycleElements.size(),
                           unscheduledCycleElements);

    for (auto taskId : unscheduled) {
      auto &pritask  = tasksMapCopy[taskId];
      auto &tensorId = pritask.name.getTensorId();

      if (pritask.name.getType() != TaskId::Type::InitTensorTask || !tensorId) {
        // Not an init tensor task
        continue;
      }

      // Replace task with dependency-free alternative
      logging::devicex::trace(
          "Using dependency-free alternative creator for task {} tensor {}",
          taskId,
          *tensorId);

      tasksMapCopy[taskId] =
          irLowering.getDependencyFreeInitTensorCreatorTask(*tensorId);

      if (!tasksMapCopy[taskId].dependsOn.empty()) {
        std::stringstream ss;
        ss << taskId << " is not a dependency-free alternative because "
           << "it has the following dependencies: ";
        for (auto it = tasksMapCopy[taskId].dependsOn.begin();
             it != tasksMapCopy[taskId].dependsOn.end();
             it++) {
          if (it != tasksMapCopy[taskId].dependsOn.begin()) {
            ss << ", ";
          }
          ss << *it;
        }
        throw error(ss.str());
      }
      pq.push(tasksMapCopy[taskId]);
      // Inject replacement task
      tasksMap[taskId] = tasksMapCopy[taskId];

      // Fallback succeeded, recurse to schedule again
      schedule(
          pq, linearisedTasks, dependantsOf, tasksMapCopy, irLowering, true);
      return;
    }
  }

  std::stringstream ss;
  ss << "different sizes of linearisedTasks (" << linearisedTasks.size()
     << ") and actual tasks (" << dependantsOf.size() << ").";
  ss << "\n tasks not in linearisedTasks:\n";

  for (auto taskId : unscheduled) {
    ss << taskId << "   [ ";
    if (tasksMap.find(taskId) != tasksMap.end()) {
      auto &depends = tasksMap.at(taskId).dependsOn;
      ss << logging::join(depends.begin(), depends.end(), ", ");
    } else {
      ss << "\n xxxxx ";
    }
    ss << ']' << "\n\n";
  }
  throw error(ss.str());
}

// this function will reorder v_tasks so that there are no dependency breakages.
std::vector<PriTask>
PriTasks::getLinearised(std::set<DependencyType> dependencies,
                        popx::IrLowering &irLowering,
                        bool allowFallback) {
  std::priority_queue<PriTask> pq;
  std::unordered_map<TaskId, std::set<TaskId>> dependantsOf;
  std::vector<PriTask> linearisedTasks;

  auto tasksMapCopy = tasksMap;

  std::set<DependencyType> removeDepTypes = {DependencyType::Output,
                                             DependencyType::Scheduler,
                                             DependencyType::Tensor};

  for (auto &x : tasksMapCopy) {
    dependantsOf[x.first] = {};
    for (auto depType : removeDepTypes) {
      if (dependencies.find(depType) == dependencies.end()) {
        x.second.removeDep(depType);
      }
    }
  }

  for (auto &x : tasksMapCopy) {
    auto name = x.first;
    auto task = x.second;
    // if a task has no dependencies, it can go directly into the queue
    if (task.dependsOn.size() == 0) {
      pq.push(task);
    }
    // otherwise, its parent dependencies are told of the dependence so that
    // when they have entered the task list, the dependency can be removed.
    else {
      for (auto &parent : task.dependsOn) {
        if (dependencies.find(parent.getType()) != dependencies.end()) {
          for (auto &taskId : parent.getTaskIds()) {
            if (dependantsOf.find(taskId) == dependantsOf.end()) {
              std::stringstream ss;
              ss << "In first step of building linearised priorities "
                 << "There is a task named " << name << " which claims to"
                 << " depend on " << taskId << " but there is no recorded task "
                 << taskId << ".";
              throw error(ss.str());
            }
            dependantsOf[taskId].insert(name);
          }
        }
      }
    }
  }

  // Perform the scheduling
  schedule(pq,
           linearisedTasks,
           dependantsOf,
           tasksMapCopy,
           irLowering,
           allowFallback);
  return linearisedTasks;
}

std::ostream &operator<<(std::ostream &oss, const DependencyType &dt) {
  switch (dt) {
  case (DependencyType::Output): {
    oss << "Output";
    break;
  }

  case (DependencyType::Tensor): {
    oss << "Tensor";
    break;
  }

  case (DependencyType::Scheduler): {
    oss << "Scheduler";
    break;
  }

  case (DependencyType::SubGraph): {
    oss << "SubGraph";
    break;
  }
  }
  return oss;
}

std::ostream &operator<<(std::ostream &oss, const PriTaskDependency &ptd) {
  oss << "PriTaskDependency[";
  oss << ptd.getType() << " ";
  oss << "(";
  size_t j = 0;
  for (auto taskId : ptd.getTaskIds()) {
    oss << taskId << ((j < ptd.getTaskIds().size() - 1) ? " || " : "");
    ++j;
  }
  oss << ") ";
  oss << "]";
  return oss;
}

} // namespace popart
