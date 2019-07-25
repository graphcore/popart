#include <algorithm>

#include <queue> // we use a priority_queue
#include <sstream>
#include <popart/error.hpp>
#include <popart/pritask.hpp>

namespace popart {

PriTask::PriTask(double p,
                 TaskId n,
                 const std::vector<TaskId> &d,
                 const std::function<void()> &f_)
    : priority(p), name(n), dependsOn(d), f(f_) {}

void PriTask::removeDep(const TaskId &dep) {
  std::vector<TaskId> newDependsOn;
  newDependsOn.reserve(dependsOn.size());
  for (auto &x : dependsOn) {
    if (x != dep) {
      newDependsOn.push_back(x);
    }
  }
  if (newDependsOn.size() + 1 != dependsOn.size()) {
    throw error(
        "failed to remove dependency '{}' for PriTask '{}'.", dep, name);
  }
  dependsOn = newDependsOn;
}

bool operator<(const PriTask &a, const PriTask &b) {
  return (a.priority < b.priority); // we want highest priority off first
}

void PriTasks::add(const PriTask &t) {
  if (tasksMap.find(t.name) != tasksMap.end()) {
    throw error("already encountered name {} in tasks", t.name);
  }
  tasksMap[t.name] = t;
}

bool PriTasks::contains(const TaskId &taskId) {
  return tasksMap.find(taskId) != tasksMap.end();
}

// this function will reorder v_tasks so that there are no dependency breakages.
std::vector<PriTask> PriTasks::getLinearised() const {
  std::priority_queue<PriTask> pq;
  std::unordered_map<TaskId, std::vector<TaskId>> dependentsOf;
  std::vector<PriTask> linearisedTasks;

  auto tasksMapCopy = tasksMap;

  for (auto &x : tasksMapCopy) {
    dependentsOf[x.first] = {};
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
        if (dependentsOf.find(parent) == dependentsOf.end()) {
          std::stringstream ss;
          ss << "In first step of building linearised priorities "
             << "There is a task named " << name << " which claims to"
             << " depend on " << parent << " but there is no recorded task "
             << parent << ".";
          throw error(ss.str());
        }
        // weird how you can just do this even if parent is not yet in map
        dependentsOf[parent].push_back(name);
      }
    }
  }

  // while there are dependency-free tasks which have not been added
  while (!pq.empty()) {
    // 1) add the lowest (highest) priority dep-free task
    linearisedTasks.push_back(pq.top());
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
    for (auto &X : dependentsOf) {
      bool present = false;
      auto parent  = X.first;
      for (auto &t : linearisedTasks) {
        if (parent == t.name) {
          present = true;
        }
      }
      if (present == false) {
        ss << parent << "   ( ";
        if (tasksMap.find(parent) != tasksMap.end()) {
          for (auto &dep : tasksMap.at(parent).dependsOn) {
            ss << dep << ' ';
          }
        } else {
          ss << " xxxxx ";
        }
        ss << ')' << '\n';
      }
    }
    throw error(ss.str());
  }

  // ok, all tasks linearised.
  return linearisedTasks;
}

} // namespace popart
