// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RINSEMATCHERBASE_HPP
#define GUARD_NEURALNET_RINSEMATCHERBASE_HPP

#include <map>
#include <vector>

#include "isomorphic.hpp"
#include "match.hpp"
#include "popart/subgraph/subgraphnames.hpp"
#include "subgraphutil.hpp"

namespace fwtools {
namespace subgraph {

// Repeated Isomorphic Non-overlapping Sequence Matcher
// R........i..........n...............se.......Matcher

template <typename T> class RinseMatcherBase {
public:
  RinseMatcherBase(const std::vector<T *> &s) : schedule(s) {
    for (int i = 0; i < schedule.size(); ++i) {
      schedule_index[schedule[i]] = i;
    }
  }

  // the schedule passed into the constructor
  // will be stored as a member
  const std::vector<T *> &schedule;

  // At what index in schedule does a T* appear?
  std::map<T *, int> schedule_index;

public:
  // partition Matches into its distinct isomorphisms
  std::vector<Match> separateMultipleMatchesByIsomorphism(
      const std::vector<Match> &matches) const {
    return separateMultipleMatches(
        matches, [this](int l, const std::vector<Start> &s0s, Start s1) {
          return RinseMatcherBase<T>::isoFunc(l, s0s, s1);
        });
  }

  // partition a Match into its distinct isomorphisms
  std::vector<Match>
  separateSingleMatchByIsomorphism(const Match &match) const {
    return separateSingleMatch(
        match, [this](int l, const std::vector<Start> &s0s, Start s1) {
          return RinseMatcherBase<T>::isoFunc(l, s0s, s1);
        });
  }

private:
  bool isoFunc(int l, const std::vector<Start> &s0s, Start s1) const {
    // s0s are all isomorphic to each other.
    // We are testing : are all the sub-graphs
    // with a start in s0s also isomorphic to the sub-graph starting at s1?
    // Case 0: if there are no sub-graphs from s0s, this is trivially true
    if (s0s.size() == 0) {
      return true;
    }
    // Case 1: isomorphism is transitive (a=b, b=c => a=c), so we only
    // need to compare to the first sub-graph
    return areIsomorphic(l, s0s[0], s1, schedule, schedule_index);
  }
};

} // namespace subgraph
} // namespace fwtools

#endif
