// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_AUPGRPAH_PRUNE_PRUNEMATCHES_HPP
#define GUARD_NEURALNET_AUPGRPAH_PRUNE_PRUNEMATCHES_HPP

#include "match.hpp"
#include <algorithm>
#include <vector>

namespace fwtools {
namespace subgraph {
namespace prune {

template <typename T, class Evaluator>
std::vector<Match> pruneMatches(const std::vector<Match> &inMatches,
                                const std::vector<T *> &schedule,
                                float threshold) {

  Evaluator f;

  std::map<T *, int> schedule_index;
  for (int i = 0; i < schedule.size(); ++i) {
    schedule_index[schedule[i]] = i;
  }

  // For each match m in inMatches, take either m or some sub-sequence of it
  std::set<Match> pruned;

  // inMatches, sorted from shortest to longest
  std::vector<Match> inMatchesIncreasing = inMatches;
  std::sort(inMatchesIncreasing.begin(),
            inMatchesIncreasing.end(),
            [](const Match &a, const Match &b) { return a.length < b.length; });

  // as we add to "pruned", we keep track of what positions have been outlined
  std::vector<bool> isInternal(schedule.size(), false);
  std::vector<bool> isFirst(schedule.size(), false);
  std::vector<bool> isLast(schedule.size(), false);

  for (auto &m : inMatchesIncreasing) {
    // initialize best value as full sub-sequence
    int minStart = m.starts[0];
    int minEnd   = m.starts[0] + m.length;
    float minAdjustedValue =
        f.value(minStart, minEnd, schedule, schedule_index);

    // consider all sub-sequences of the match, computing the value
    for (int start = m.starts[0]; start < m.starts[0] + m.length; ++start) {
      for (int end = start + 1; end <= m.starts[0] + m.length; ++end) {

        // valid start (not overlapping with covered)
        bool goodStart =
            (!isInternal[start] && // another Match is here, and not its start
             !isLast[start]        // another Match is here, and not its start
            );

        // valid end (not overlapping with covered)
        bool goodEnd =
            (!isInternal[end - 1] && // another match is here, and not its end
             !isFirst[end]           // another match is here, and not its end
            );

        // we don't need to check for isomorphism, sub-sequences always are iso.
        if (goodStart && goodEnd) {

          float value = f.value(start, end, schedule, schedule_index);
          if (value > minAdjustedValue) {
            minAdjustedValue = value;
            minStart         = start;
            minEnd           = end;
          }
        }
      }
    }

    std::vector<Start> newStarts(m.starts);
    for (auto &x : newStarts) {
      x += minStart - m.starts[0];
    }

    Match newMatch(newStarts, minEnd - minStart);
    setValue(newMatch, schedule);
    if (newMatch.getDiscountedValue() > threshold) {
      pruned.insert(newMatch);
    }

    if (newMatch.length > 1) {
      for (auto s0 : newStarts) {
        for (int64_t d = 1; d <= newMatch.length - 2; ++d) {
          isInternal[s0 + d] = true;
        }
        isFirst[s0]                      = true;
        isLast[s0 + newMatch.length - 1] = true;
      }
    }
  }

  std::vector<Match> vPruned;
  vPruned.reserve(pruned.size());
  for (const auto &m : pruned) {
    vPruned.push_back(m);
  }
  return vPruned;
}

} // namespace prune
} // namespace subgraph
} // namespace fwtools

#endif
