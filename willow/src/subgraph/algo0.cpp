// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/subgraph/algo0.hpp>

namespace fwtools {
namespace subgraph {
namespace algo0 {

std::vector<Match> getRepeatedIntSequences(const std::vector<int> &intSched) {

  auto n_nodes = intSched.size();

  // series = [0, 1, 2, ... n_nodes)
  std::vector<int> series(n_nodes, 0);
  std::iota(series.begin(), series.end(), 0);

  // each map in sub_strings contains,
  // key: FIRST index where the sub-string starts
  // values: ALL indices where the sub-string starts
  // the length of the sub-strings is the index of the map
  // in the vector, so sub_strings[5] contains sub-strings of
  // length 5
  std::vector<std::map<Start, std::vector<Start>>> sub_strings;

  // initialized for length 0:
  int current_length = 0;
  sub_strings.push_back({{0, series}});

  // while there are sub-strings at current_length,
  while (sub_strings.back().size() != 0) {
    ++current_length;
    std::map<Start, std::vector<Start>> current_sub_strings{};
    for (auto first_starts : sub_strings.back()) {
      std::map<int, std::vector<Start>> by_end_type;
      for (auto start : first_starts.second) {
        int end = start + current_length - 1;
        if (end < n_nodes) {
          auto id    = intSched[end];
          auto found = by_end_type.find(id);
          if (found == by_end_type.end()) {
            by_end_type[id] = {start};
          } else {
            found->second.push_back(start);
          }
        }
      }
      for (auto &t_starts : by_end_type) {
        auto &starts = t_starts.second;
        if (starts.size() > 1) {
          current_sub_strings[starts[0]] = starts;
        }
      }
    }
    sub_strings.push_back(current_sub_strings);
  }

  std::vector<Match> matches;
  for (int l = 1; l < sub_strings.size(); ++l) {
    for (auto &first_starts : sub_strings[l]) {
      matches.push_back({first_starts.second, l});
    }
  }
  return matches;
}

std::vector<Match> getFinalMatches(const std::vector<Match> &matches,
                                   float threshold,
                                   int schedule_size) {

  std::vector<Match> descending_filter_matches{};

  auto isGoodMatch =
      [threshold, &descending_filter_matches](const Match &candidate_match) {
        // does candidate_match overlap with any
        // of the matches in descending_filter_matches so far?
        for (auto &accepted_match : descending_filter_matches) {
          if (candidate_match.crosses(accepted_match)) {
            return false;
          }
          if (accepted_match.subsumes(candidate_match)) {
            return false;
          }

          if (accepted_match.intersects(candidate_match) &&
              !accepted_match.fitsCleanly(candidate_match)) {
            return false;
          }
        }

        if (candidate_match.getValue() < threshold) {
          return false;
        }

        return true;
      };

  for (auto &candidate_match : matches) {
    if (isGoodMatch(candidate_match)) {
      descending_filter_matches.push_back(candidate_match);
    }
  }

  // Final removal of matches with low incremental value,
  // done in increasing order of match values.
  std::reverse(descending_filter_matches.begin(),
               descending_filter_matches.end());

  // descending_filter_matches is now in ascending order of value
  const std::vector<Match> dfm(descending_filter_matches);

  return applyIncrementalThreshold(dfm, schedule_size, threshold);
}

} // namespace algo0
} // namespace subgraph
} // namespace fwtools
