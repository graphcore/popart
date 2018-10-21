#ifndef GUARD_WILLOW_PRITASK_HPP
#define GUARD_WILLOW_PRITASK_HPP

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace willow {

/** A Pritask is a task which has a priority and a set of dependent tasks. */
class PriTask {
public:
  double priority;
  // the name of this task, must be a unique identifier
  std::string name;
  // the names of tasks which MUST appear before this task in a linearisation
  std::vector<std::string> dependsOn;
  std::function<void()> f;
  PriTask() = default;
  PriTask(double p,
          std::string n,
          const std::vector<std::string> &d,
          const std::function<void()> &f_);

  // remove dep from dependsOn.
  void removeDep(const std::string &dep);
};


// returns true if "a" has lower priority than "b"
bool operator<(const PriTask &a, const PriTask &b);

class PriTasks {
public:
  std::unordered_map<std::string, PriTask> tasksMap;
  void add(const PriTask &t);

  // return the tasks in an order of descending priority as far 
  // as possible, subject to all dependencies being satisfied.
  // the algorithm used is yet another variant of  Kahn's algorithm
  std::vector<PriTask> getLinearised() const;
  PriTasks()  = default;

};

} // namespace willow

#endif
