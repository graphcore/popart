// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_VALIDATE_HPP_
#define POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_VALIDATE_HPP_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <popart/subgraph/isomorphic.hpp>
#include <popart/subgraph/match.hpp>

#include "popart/subgraph/subgraphnames.hpp"

template <typename T>
bool isValid(const std::vector<fwtools::subgraph::Match> &matches,
             const std::vector<T *> &schedule) {

  using namespace fwtools::subgraph;

  std::map<T *, int> schedule_index;
  for (int i = 0; i < schedule.size(); ++i) {
    schedule_index[schedule[i]] = i;
  }
  int schedule_size = schedule.size();

  // confirm isomorphisms
  // --------------------
  for (auto &m : matches) {
    if (m.starts.size() <= 1 || m.length < 1) {
      return false;
    }
    for (int start_index = 1; start_index < m.starts.size(); ++start_index) {
      if (isomorphicUntil(m.length,
                          m.starts[start_index],
                          m.starts[0],
                          schedule,
                          schedule_index) != m.length) {
        return false;
      }
    }
  }

  // confirm nesting
  // ---------------
  std::vector<Match> revMatches(matches);
  std::sort(revMatches.begin(),
            revMatches.end(),
            [](const Match &a, const Match &b) { return a.length < b.length; });
  std::vector<std::vector<const Match *>> blanket(schedule_size);

  for (int i = 0; i < revMatches.size(); ++i) {
    for (int j = 0; j < revMatches[i].starts.size(); ++j) {
      blanket[revMatches[i].starts[j]].push_back(&revMatches[i]);
    }

    for (int j = 1; j < revMatches[i].starts.size(); ++j) {
      for (int k = 0; k < revMatches[i].length; ++k) {
        if (blanket[revMatches[i].starts[0] + k] !=
            blanket[revMatches[i].starts[j] + k]) {
          std::cout << "failure for match " << revMatches[i]
                    << ", between starts " << 0 << " and " << j
                    << " with offset " << k << " : ";
          std::cout << blanket[revMatches[i].starts[0] + k].size()
                    << " != " << blanket[revMatches[i].starts[j] + k].size()
                    << std::endl;

          std::cout << "blanket 0 : " << std::endl;
          for (auto &x : blanket[revMatches[i].starts[0] + k]) {
            std::cout << *x << std::endl;
          }

          std::cout << "blanket j = " << j << " : " << std::endl;
          for (auto &x : blanket[revMatches[i].starts[j] + k]) {
            std::cout << *x << std::endl;
          }

          return false;
        }
      }
    }
  }

  // confirm no-overlapping
  // ----------------------

  return true;
}

#endif // POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_VALIDATE_HPP_
