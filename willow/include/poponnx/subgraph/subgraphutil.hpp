#ifndef GUARD_NEURALNET_SUBGRAPHS_HPP
#define GUARD_NEURALNET_SUBGRAPHS_HPP

#include "isomorphic.hpp"
#include "match.hpp"
#include "subgraphnames.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fwtools {
namespace subgraph {

std::vector<Match> applyIncrementalThreshold(const std::vector<Match> &dfm,
                                             int schedule_size,
                                             float threshold);

template <typename T>
std::vector<int> getIntSchedule(const std::vector<T *> &schedule) {
  int n_nodes = static_cast<int>(schedule.size());
  std::vector<int> intSched;
  intSched.reserve(n_nodes);
  std::map<decltype(schedule[0]->getSubgraphEquivId()), int> intMap;
  int nUniqueChars = 0;
  for (auto &t : schedule) {
    auto equivId = t->getSubgraphEquivId();
    auto found   = intMap.find(equivId);
    int mapped;
    if (found == intMap.end()) {
      intMap[equivId] = nUniqueChars;
      mapped          = nUniqueChars;
      ++nUniqueChars;
    } else {
      mapped = found->second;
    }
    intSched.push_back(mapped);
  }
  return intSched;
}

// the value of match, currently just the sum of the values of the nodes
template <class T>
void setValue(Match &match, const std::vector<T *> &schedule) {
  float value = 0;
  for (int i = 0; i < match.length; ++i) {
    value += schedule.at(match.starts[0] + i)->getSubgraphValue();
  }
  match.setValue(value);
}

template <class T>
std::vector<float> getCumVals(const std::vector<T *> &sched) {
  std::vector<float> cumVals(sched.size() + 1, 0.0f);
  for (int i = 0; i < sched.size(); ++i) {
    cumVals[i + 1] = cumVals[i] + sched.at(i)->getSubgraphValue();
  }
  return cumVals;
}

template <class T>
void setValues(std::vector<Match> &matches, const std::vector<T *> &schedule) {
  for (auto &match : matches) {
    setValue<T>(match, schedule);
  }
}

std::vector<Match> separateMultipleMatches(
    const std::vector<Match> &matches,
    const std::function<bool(int, const std::vector<Start> &, Start)> &comp);

std::vector<Match> separateSingleMatch(
    const Match &match,
    const std::function<bool(int, const std::vector<Start> &, Start)> &comp);

// if matches within a Match intersect, divide the Match
std::vector<Match> separateByOverlaps(const std::vector<Match> &matches);

bool areIntersecting(int seq_length0,
                     int seq_length1,
                     const std::vector<Start> &s0s,
                     Start s1);

// this is stronger than areIntersecting, as
// they must intersect and not be nested.
bool areCrossing(int seq_length0,
                 int seq_length1,
                 const std::vector<Start> &s0s,
                 Start s1);

} // namespace subgraph
} // namespace fwtools

#endif
