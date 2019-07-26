#include "popart/subgraph/subgraphutil.hpp"

#include <algorithm>
#include <tuple>

namespace fwtools {
namespace subgraph {

std::vector<Match> separateByOverlaps(const std::vector<Match> &matches) {
  auto nonOverlapping =
      [](int seq_length, const std::vector<Start> &s0s, Start s1) {
        return !areIntersecting(seq_length, seq_length, s0s, s1);
      };
  return separateMultipleMatches(matches, nonOverlapping);
}

// Divide a Match with incomaptible starts into
// multiple Matches with compatible starts.
std::vector<Match> separateSingleMatch(
    const Match &match,
    const std::function<bool(int, const std::vector<Start> &, Start)> &comp) {

  auto &seq_starts = match.starts;
  int seq_length   = match.length;

  std::vector<Match> local;
  for (Start start : seq_starts) {
    bool found_match = false;
    for (auto &match2 : local) {
      bool compatible = comp(seq_length, match2.starts, start);
      if (compatible) {
        match2.starts.push_back(start);
        found_match = true;
        break;
      }
    }
    if (!found_match) {
      local.push_back({{start}, seq_length});
    }
  }

  // If at the end here, local has length 1,
  // then all sequences in match were compatible
  // (isomorphic / non-overlapping).
  // If at the end, local has length = seq_length,
  // then all sequences in match incompatible

  std::vector<Match> repeateds;
  for (auto &match_2 : local) {
    if (match_2.starts.size() > 1) {
      repeateds.push_back(match_2);
    }
  }

  return repeateds;
}

std::vector<Match> separateMultipleMatches(
    const std::vector<Match> &matches,
    const std::function<bool(int, const std::vector<Start> &, Start)> &comp) {
  std::vector<Match> all_repeateds;
  for (const auto &match : matches) {
    auto repeateds = separateSingleMatch(match, comp);
    all_repeateds.insert(
        all_repeateds.end(), repeateds.begin(), repeateds.end());
  }
  return all_repeateds;
}

bool areIntersecting(int seq_length0,
                     int seq_length1,
                     const std::vector<Start> &s0s,
                     Start s1) {

  // we are checking that the interval
  //        [s1, s1 + seq_length1)
  // intersects with any of the intervals
  //        [s0, s0 + seq_length0)
  // where s0 is in s0s.

  for (auto s0 : s0s) {

    // ....xxxx..... [s0, s0 + seq_length)
    // ......xxxx... [s1, s1 + seq_length)
    if (s0 < s1 && s0 + seq_length0 > s1) {
      return true;
    }

    // the reverse of the above
    else if (s1 <= s0 && s1 + seq_length1 > s0) {
      return true;
    }
  }

  // no intersection for any s0
  return false;
}

// Where dfm is sorted in ascending order of value,
// we filter out (starting from least valuable) Matches
// whose incremental value is below threshold
std::vector<Match> applyIncrementalThreshold(const std::vector<Match> &dfm,
                                             int schedule_size,
                                             float threshold) {

  // pointers into dfm, this vector contains the starts
  // of all covering Matches. Recall we start from the
  // smallest Match. Suppose we go from
  // .....xx.......xx......www.......
  // to
  // ...aaaaa....aaaaa.....www..aaaaa.
  //
  // The blanket goes from
  // .....P........P.......Q.........
  // to
  // ...R........R.........Q....R....
  //
  // where . is the nullptr,
  //       P, Q, R are pointers to matches

  std::vector<const Match *> blanket(schedule_size, nullptr);
  std::vector<Match> final_matches;

  for (int i = 0; i < dfm.size(); ++i) {
    Start start0    = dfm[i].starts[0];
    int len         = dfm[i].length;
    float vChildren = 0.0f;
    for (int j = start0; j < start0 + len; ++j) {
      if (blanket[j]) {
        vChildren += blanket[j]->getValue();
      }
    }

    if (dfm[i].getValue() - vChildren > threshold) {
      final_matches.push_back(dfm[i]);

      for (Start start : dfm[i].starts) {

        for (int j = start; j < start + len; ++j) {
          blanket[j] = nullptr;
        }
        blanket[start] = &dfm[i];
      }
    }
  }

  std::reverse(final_matches.begin(), final_matches.end());
  return final_matches;
}

} // namespace subgraph
} // namespace fwtools
