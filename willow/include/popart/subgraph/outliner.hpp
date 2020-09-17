// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RINSEMATCHER_OUTLINER_HPP
#define GUARD_NEURALNET_RINSEMATCHER_OUTLINER_HPP

#include "algo0.hpp"
#include "algo1.hpp"
#include <popart/logging.hpp>

namespace fwtools {
namespace subgraph {

enum class OutlinerAlgorithm {

  // Bottom-up
  // Generate all sub-string matches in the schedule (O(N^2)), then
  // 1) for all matches, if they are not isomorphic, partition into isomorphisms
  // 2) for all matches, if they are overlapping, partition into non-overlapping
  // 3) starting with accepted = [], for all matches starting with the most
  // valuable, if compatible with already accepted matches, accept it. Best
  // case: linear (if only small matches). Worst case : cubic : O(N^2)
  // isomorphism tests, each of which is O(N)
  ALGO0,

  // Top-down
  // Get all internal nodes of the suffix tree (construced in O(N) time with
  // Ukkonen's algorithm) and put them in a priority queue. While the queue
  // is not empty: take most valuable Match and if it is {not isomorphic,
  // overlapping, crosses with already accepted} then spawn smaller Matches from
  // it. Otherwise accept it. Best case : O(N) if a big match is found which
  // empties the queue. Worst case when all isomorphic: quadratic. Worst case
  // when few isomorphisms cubic (see T7779)
  ALGO1
};

// ALGO1
OutlinerAlgorithm getDefaultOutlinerAlgorithm();

template <typename T>

// Repeated isomorphic non-overlapping sequences
// R....... i......... n.............. se....... matching function
std::vector<Match>
getRinseMatches(const std::vector<T *> &schedule,
                const std::vector<std::pair<size_t, size_t>> &sequences,
                float threshold,
                float sequenceBreakCost,
                OutlinerAlgorithm algo) {

  switch (algo) {

  case OutlinerAlgorithm::ALGO0: {
    using namespace algo0;
    RinseMatcherAlgo0<T> matcher(schedule);
    auto matches = matcher.getRepeatedSequences();
    matches      = matcher.separateMultipleMatchesByIsomorphism(matches);
    matches      = separateByOverlaps(matches);
    setValues(matches, schedule);
    std::sort(matches.rbegin(), matches.rend());
    popart::logging::trace("[getRinseMatches] pre final matches: {}",
                           matches.size());

    matches =
        getFinalMatches(matches, threshold, static_cast<int>(schedule.size()));
    popart::logging::trace("[getRinseMatches] final matches: {}",
                           matches.size());

    return matches;
  }

  case OutlinerAlgorithm::ALGO1: {
    using namespace algo1;
    Algo1<T> algo1(schedule, sequences, sequenceBreakCost);
    algo1.init();
    auto acc = algo1.getPreThresholded();

    popart::logging::trace("[getRinseMatches] #matches pre threshold: {}",
                           acc.size());

    // run the blanket algorithm, to remove matches with only incremental
    // value
    acc = applyIncrementalThreshold(
        acc, static_cast<int>(schedule.size()), threshold);

    popart::logging::trace("[getRinseMatches] #matches post threshold: {}",
                           acc.size());

    return acc;
  }
  }
  return std::vector<Match>{};
}

} // namespace subgraph
} // namespace fwtools

#endif
