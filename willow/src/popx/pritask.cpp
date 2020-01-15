#include <algorithm>

#include <queue> // we use a priority_queue
#include <sstream>
#include <popart/error.hpp>
#include <popart/popx/pritask.hpp>

namespace popart {

PriTask::PriTask(double p,
                 TaskId n,
                 const std::vector<std::pair<TaskId, DependencyType>> &d,
                 const std::function<SequenceMap()> &f_)
    : priority(p), name(n), dependsOn(d), f(f_) {}

// Remove all dependencies of TaskId dep
void PriTask::removeDep(const TaskId &dep) {
  std::vector<std::pair<TaskId, DependencyType>> newDependsOn;
  newDependsOn.reserve(dependsOn.size());
  for (auto &x : dependsOn) {
    if (x.first != dep) {
      newDependsOn.push_back(x);
    }
  }
  // Multiple times the same dependency possible, with different type
  if (newDependsOn.size() >= dependsOn.size()) {
    throw error(
        "failed to remove dependency '{}' for PriTask '{}'.", dep, name);
  }
  dependsOn = newDependsOn;
}

void PriTask::removeDep(DependencyType type) {
  std::vector<std::pair<TaskId, DependencyType>> newDependsOn;
  newDependsOn.reserve(dependsOn.size());
  for (auto &x : dependsOn) {
    if (x.second != type) {
      newDependsOn.push_back(x);
    }
  }
  dependsOn = newDependsOn;
}

std::set<TaskId>
PriTask::getDependenciesOfTypes(std::set<DependencyType> depTypes) {
  std::set<TaskId> taskIds;
  for (auto &dep : dependsOn) {
    if (depTypes.find(dep.second) != depTypes.end()) {
      taskIds.insert(dep.first);
    }
  }
  return taskIds;
}

bool operator<(const PriTask &a, const PriTask &b) {
  return (a.priority < b.priority); // we want highest priority off first
}

void PriTasks::add(const PriTask &t) {
  if (tasksMap.find(t.name) != tasksMap.end()) {
    throw error("already encountered name {} in tasks", t.name);
  }

  std::ostringstream oss;
  oss << "Adding Pritask " << t.name << " <- [ ";
  for (auto dep : t.dependsOn) {
    oss << dep.first << ' ';
  }
  oss << "]";
  logging::devicex::debug(oss.str());

  for (auto x : t.dependsOn) {
    auto found = tasksMap.find(x.first);
    if (found != tasksMap.end()) {
      if (std::find(found->second.dependsOn.begin(),
                    found->second.dependsOn.end(),
                    std::make_pair(t.name, DependencyType::OUTPUT)) !=
          found->second.dependsOn.end()) {
        throw error("circular PriTask dependency " + x.first + " <-> " +
                    t.name);
      }
    }
  }
  tasksMap[t.name] = t;
}

bool PriTasks::contains(const TaskId &taskId) {
  return tasksMap.find(taskId) != tasksMap.end();
}

// this function will reorder v_tasks so that there are no dependency breakages.
std::vector<PriTask>
PriTasks::getLinearised(std::set<DependencyType> dependencies) const {
  std::priority_queue<PriTask> pq;
  std::unordered_map<TaskId, std::set<TaskId>> dependentsOf;
  std::vector<PriTask> linearisedTasks;

  auto tasksMapCopy = tasksMap;

  std::set<DependencyType> removeDepTypes = {DependencyType::OUTPUT,
                                             DependencyType::SCHEDULER,
                                             DependencyType::TENSOR};

  for (auto &x : tasksMapCopy) {
    dependentsOf[x.first] = {};
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
        if (dependencies.find(parent.second) != dependencies.end()) {
          if (dependentsOf.find(parent.first) == dependentsOf.end()) {
            std::stringstream ss;
            ss << "In first step of building linearised priorities "
               << "There is a task named " << name << " which claims to"
               << " depend on " << parent.first
               << " but there is no recorded task " << parent.first << ".";
            throw error(ss.str());
          }
          // weird how you can just do this even if parent is not yet in map
          dependentsOf[parent.first].insert(name);
        }
      }
    }
  }

  // while there are dependency-free tasks which have not been added
  while (!pq.empty()) {
    // 1) add the lowest (highest) priority dep-free task
    linearisedTasks.push_back(tasksMap.at(pq.top().name));
    auto &parent = linearisedTasks.back().name;
    pq.pop();
    // update the dependencies of child tasks, pushing child tasks
    // onto the priority queue if they become dep-free.
    for (auto &child : dependentsOf.at(parent)) {
      tasksMapCopy[child].removeDep(parent);
      if (tasksMapCopy[child].dependsOn.size() == 0) {
        pq.push(tasksMapCopy[child]);
      }
    }
  }

  // confirm that the linearisedTasks contains all the tasks.
  // circular dependencies will prevent this.
  if (linearisedTasks.size() != dependentsOf.size()) {
    std::stringstream ss;
    ss << "different sizes of linearisedTasks (" << linearisedTasks.size()
       << ") and actual tasks (" << dependentsOf.size() << ").";
    ss << "\n tasks not in linearisedTasks:\n";
    for (auto &x : dependentsOf) {
      bool present = false;
      auto parent  = x.first;
      for (auto &t : linearisedTasks) {
        if (parent == t.name) {
          present = true;
        }
      }
      if (present == false) {
        ss << parent << "   [ ";
        if (tasksMap.find(parent) != tasksMap.end()) {
          for (auto &dep : tasksMap.at(parent).dependsOn) {
            ss << "\n       " << dep.first << " ("
               << static_cast<int>(dep.second) << ") ";
          }
        } else {
          ss << "\n xxxxx ";
        }
        ss << ']' << "\n\n";
      }
    }
    throw error(ss.str());
  }

  // ok, all tasks linearised.
  return linearisedTasks;
}

} // namespace popart
