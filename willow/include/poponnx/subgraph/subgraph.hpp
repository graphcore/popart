#ifndef GUARD_NEURALNET_SUBGRAPHS_HPP
#define GUARD_NEURALNET_SUBGRAPHS_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fwtools {
namespace subgraph {

// To compare nodes in the graph for equivalence, use a std::string
using EquivId = std::string;

// The position in the schedule that a sequence (sub-graph) starts at
using Start = int;

// The input and output indices of a node
using InIndex  = int;
using OutIndex = int;

class Match {
public:
  Match(const std::vector<Start> &s, int l) : starts(s), length(l) {}

  // the indices in a schedule at which the identical sequences start at
  std::vector<Start> starts;

  // the length of the identical sequences
  int length;

  bool operator<(const Match &other) const {
    // the length of the sequence (sub-graph) takes
    // highest priority in the comparison
    if (length != other.length) {
      return length < other.length;
    }
    return starts < other.starts;
  }

  bool operator==(const Match &other) const {
    return length == other.length && starts == other.starts;
  }
};

// -----------------------------------------------
// The key function for sub-graph matching.
//--------------------------------------------------------------
template <typename T> std::vector<Match> getMatches(const std::vector<T *> &s);
//--------------------------------------------------------------
// (the RinseMatcher class is for power-users only)
//-------------------------------------------------

std::ostream &operator<<(std::ostream &stream, const Match &match) {
  stream << match.length << " | [";
  if (match.starts.size() != 0) {
    stream << match.starts[0];
  }
  for (int i = 1; i < match.starts.size(); ++i) {
    stream << ", " << match.starts[i];
  }
  stream << "]";
  return stream;
}

// Each edge into the sequence (sub-graph) is described by
// 1) an offset "delta" from the beginning of the sequence
// 2) the index "inIndex" at which the edge enters
struct InternalConsumer {
  InternalConsumer(int d_, InIndex i_) : delta(d_), inIndex(i_) {}
  int delta;
  InIndex inIndex;

  bool operator==(const InternalConsumer &other) const {
    return (delta == other.delta && inIndex == other.inIndex);
  }

  bool operator!=(const InternalConsumer &other) const {
    return !(*this == other);
  }
};

// Repeated Isomorphic Non-overlapping Sequence Matcher
// R........i..........n...............se.......Matcher
//
// TODO template should take an optional template parameter for comparison
//
// template paramter T might be poponnx::Op, or
// const tf::Op (is that the tensorflow class?)
template <typename T> class RinseMatcher {
public:
  RinseMatcher(const std::vector<T *> &s);

  // the schedule passed into the constructor will be stored as a member
  const std::vector<T *> &schedule;

  // At what index in schedule does a T* appear?
  std::map<T *, int> schedule_index;

public:
  // partition a Match into its distinct isomorphisms
  std::vector<Match>
  separateByIsomorphism(const std::vector<Match> &matches) const;

  // sort the Matches by their value
  std::vector<Match> getPrioritized(const std::vector<Match> &matches) const;

  // get the final Matches from the prioritised vector of Matches,
  // not taking Matches which are subsumed by, or those that cross over
  // already accepted Matches
  std::vector<Match>
  getFinalMatches(const std::vector<Match> &prioritised) const;

  // if matches within a Match intersect, divide the Match
  std::vector<Match>
  separateByOverlaps(const std::vector<Match> &matches) const;

  // sub-string matching for repeated sequences
  std::vector<Match> getRepeatedSequences() const;

private:
  // used by separateByOverlaps and separateByIsomorphism
  std::vector<Match>
  separate(const std::vector<Match> &matches,
           const std::function<bool(int, const std::vector<Start> &, Start)>
               &comp) const;

  // the value of match, currently just the sum of the values of the nodes
  float getValue(const Match &match) const {
    float value = 0;
    for (int i = 0; i < match.length; ++i) {
      value += schedule.at(match.starts[0] + i)->getValue();
    }
    return value;
  }

  bool areIsomorphic(int seq_length, Start s0, Start s1) const;

  // how far ahead of start is node in the schedule?
  int relativeToStart(T *node, Start start) const;

  // identifying a connection (a tensor for neural nets)
  // poponnx : (Op * creator, output index of creator, TensorId)
  // tf : similar, but the string can be empty
  using Input = std::tuple<T *, OutIndex, std::string>;
};

template <typename T>
RinseMatcher<T>::RinseMatcher(const std::vector<T *> &s) : schedule(s) {
  for (int i = 0; i < schedule.size(); ++i) {
    schedule_index[schedule[i]] = i;
  }
}

template <typename T>
int RinseMatcher<T>::relativeToStart(T *node, Start start) const {
  auto index = schedule_index.at(node);
  return index - start;
}

static bool areIntersecting(int seq_length0,
                            int seq_length1,
                            const std::vector<Start> &s0s,
                            Start s1) {
  // we are checking that the interval
  //        [s1, s1 + seq_length1)
  // intersects with any of the intervals
  //        [s0, s0 + seq_length0)
  // where s0 is in s0s.
  for (auto s0 : s0s) {
    // ....XXXX..... [s0, s0 + seq_length)
    // ......XXXX... [s1, s1 + seq_length)
    if (s0 < s1 && s0 + seq_length0 > s1) {
      return true;
    }

    else if (s1 <= s0 && s1 + seq_length1 > s0) {
      return true;
    }
  }

  // no intersection for any s0
  return false;
}

// this is stronger than areIntersecting, as
// they must intersect and not be nested.
static bool areCrossing(int seq_length0,
                        int seq_length1,
                        const std::vector<Start> &s0s,
                        Start s1) {

  for (auto s0 : s0s) {
    auto e0 = s0 + seq_length0;
    auto e1 = s1 + seq_length1;
    if (s0 < s1 && e0 > s1 && e0 < e1) {
      return true;
    }

    // the reverse of the case above:
    else if (s1 < s0 && e1 > s0 && e1 < e0) {
      return true;
    }

    // note that the case s1 == s0 always returns false
  }

  // no intersection for any s0
  return false;
}

template <typename T>
std::vector<Match>
RinseMatcher<T>::separateByOverlaps(const std::vector<Match> &matches) const {

  auto nonOverlapping =
      [](int seq_length, const std::vector<Start> &s0s, Start s1) {
        return !areIntersecting(seq_length, seq_length, s0s, s1);
      };

  return separate(matches, nonOverlapping);
}

template <typename T>
std::vector<Match> RinseMatcher<T>::separate(
    const std::vector<Match> &matches,
    const std::function<bool(int, const std::vector<Start> &, Start)> &comp)
    const {

  std::vector<Match> repeateds;
  for (const auto &match : matches) {

    auto &seq_starts = match.starts;
    int seq_length   = match.length;
    // if at the end, local has length 1, then all sequences in
    // match were (isomorphic / non-overlapping).
    // If at the end, local has
    // length = seq_length, then all sequences in
    // match were (isomorphically different / overlapping)

    std::vector<Match> local;
    for (Start start : seq_starts) {
      bool foundMatch = false;
      for (auto &match2 : local) {
        if (comp(seq_length, match2.starts, start)) {
          match2.starts.push_back(start);
          foundMatch = true;
          break;
        }
      }
      if (!foundMatch) {
        local.push_back({{start}, seq_length});
      }
    }
    for (auto &match : local) {
      if (match.starts.size() > 1) {
        repeateds.push_back(match);
      }
    }
  }
  return repeateds;
}

template <typename T>
std::vector<Match>
RinseMatcher<T>::getPrioritized(const std::vector<Match> &matches) const {

  // apend values to the matches, in preparation for sorting by value
  std::vector<std::tuple<float, Match>> valued;
  for (auto &m : matches) {
    valued.emplace_back(std::tuple<float, Match>(getValue(m), m));
  }

  std::sort(
      valued.begin(),
      valued.end(),
      [](const std::tuple<float, Match> &a, const std::tuple<float, Match> &b) {
        // if the float values are different, use them
        if (std::get<0>(a) < std::get<0>(b) ||
            std::get<0>(a) > std::get<0>(b)) {
          return std::get<0>(a) < std::get<0>(b);
        }

        // otherwise, use the number of sequences matched,
        return std::get<1>(a).starts.size() < std::get<1>(b).starts.size();
      });
  std::vector<Match> prioritized;
  prioritized.reserve(valued.size());

  // set in order of decreasing value
  for (auto x = valued.rbegin(); x != valued.rend(); ++x) {
    prioritized.push_back(std::get<1>(*x));
  }

  return prioritized;
}

template <typename T>
std::vector<Match>
RinseMatcher<T>::getFinalMatches(const std::vector<Match> &matches) const {
  std::vector<Match> final_matches{};

  auto isGoodMatch = [&final_matches](const Match &candidate_match) {
    // does candidate_match overlap with any
    // of the matches in final_matches so far?
    for (auto &accepted_match : final_matches) {
      for (auto candidate_start : candidate_match.starts) {
        if (areCrossing(accepted_match.length,
                        candidate_match.length,
                        accepted_match.starts,
                        candidate_start)) {
          return false;
        }
      }
    }

    // is candidate_match trivial (subsumed by a match in final_matches)?
    // Example of subsuming:
    //   ++++      ++++     ++++
    // ******    ******   ******
    // --------------------------> schedule
    //
    // if ****** has already been included in final_matches, we
    // do not want ++++ in final_matches.
    // Another example, which is NOT subsuming:
    //
    //   ++  ++     ++  ++    ++  ++
    //   *******    *******   *******
    //---------------------------------> schedule
    //
    // in the case above, we say that ++ is not subsumed by *******
    // as it appears more than once in *******.

    for (auto &accepted_match : final_matches) {
      if (accepted_match.starts.size() == candidate_match.starts.size()) {

        auto candidate_starts = candidate_match.starts;
        auto accepted_starts  = accepted_match.starts;
        std::sort(candidate_starts.begin(), candidate_starts.end());
        std::sort(accepted_starts.begin(), accepted_starts.end());

        bool subsumed = true;
        for (int i = 0; i < accepted_starts.size(); ++i) {
          // if the candidate starts before, or ends after the one already
          // accepted, it is not subsumed
          if (candidate_starts[i] < accepted_starts[i] ||
              candidate_starts[i] + candidate_match.length >
                  accepted_starts[i] + accepted_match.length) {
            subsumed = false;
            break;
          }
        }

        // if subsumed, it is not a good match
        if (subsumed) {
          return false;
        }
      }
    }
    return true;
  };

  for (auto &candidate_match : matches) {
    if (isGoodMatch(candidate_match)) {
      final_matches.push_back(candidate_match);
    }
  }
  return final_matches;
}

template <typename T>
std::vector<Match> RinseMatcher<T>::separateByIsomorphism(
    const std::vector<Match> &matches) const {
  auto areIso = [this](int l, const std::vector<Start> &s0s, Start s1) {
    // we are testing : are all the sub-graphs
    // with a start in s0s isomorphic to the sub-graph starting at s1.
    // if there are no sub-graphs from s0s, this is trivially true
    if (s0s.size() == 0) {
      return true;
    }
    // otherwise, as isomorphism is an identity-like property, we only
    // need to compare to the first sub-graph
    return areIsomorphic(l, s0s[0], s1);
  };
  return separate(matches, areIso);
}

template <typename T>
std::vector<Match> RinseMatcher<T>::getRepeatedSequences() const {

  int n_nodes = static_cast<int>(schedule.size());

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
      std::map<EquivId, std::vector<Start>> by_end_type;
      for (auto start : first_starts.second) {
        int end = start + current_length - 1;
        if (end < n_nodes) {
          auto id    = schedule[end]->getEquivId();
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

template <typename T>
bool RinseMatcher<T>::areIsomorphic(int seq_length, Start s0, Start s1) const {

  // the producers of the inputs for a node in a sub-graph
  // are either in the sub-graph or not. When
  // 1) they are in the sub-graph, the corresponding producers
  //    in sub-graph 0 and sub-graph 1 must be at the same index
  //    relative to the start of the sub-graph, and the output
  //    index of the producers must be the same
  // 2) they aren't in the sub-graph, the corresponding consumers
  //    for the 2 sub-graphs must have identical consumers, at identical
  //    consumer InIndexs, for identical OutIndexs

  // case 2 : external producers. These maps should be identical
  std::map<Input, std::vector<InternalConsumer>> externProds0;
  std::map<Input, std::vector<InternalConsumer>> externProds1;

  for (int delta = 0; delta < seq_length; ++delta) {
    auto &t0 = schedule[s0 + delta];
    auto &t1 = schedule[s1 + delta];
    if (t0->getInIndices() != t1->getInIndices()) {
      // this should actually be an error, as they shouldn't have
      // been returned as equivalent
      return false;
    }

    auto &ins0 = t0->getInputs();
    auto &ins1 = t1->getInputs();

    for (auto inIndex : t0->getInIndices()) {
      auto &in0 = ins0.at(inIndex);
      auto &in1 = ins1.at(inIndex);

      auto prod0 = std::get<0>(in0);
      auto out0  = std::get<1>(in0);
      auto rel0  = relativeToStart(prod0, s0);

      auto prod1 = std::get<0>(in1);
      auto out1  = std::get<1>(in1);
      auto rel1  = relativeToStart(prod1, s1);

      if (rel0 < 0 && rel1 < 0) {
        // both are external
        // 0
        auto found0 = externProds0.find(in0);
        if (found0 == externProds0.end()) {
          externProds0[in0] = {{delta, inIndex}};
        } else {
          externProds0[in0].push_back({delta, inIndex});
        }
        // 1
        auto found1 = externProds1.find(in1);
        if (found1 == externProds1.end()) {
          externProds1[in1] = {{delta, inIndex}};
        } else {
          externProds1[in1].push_back({delta, inIndex});
        }

        // we check if they are still isomorphic
        // with this external input now included
        if (externProds0[in0] != externProds1[in1]) {
          return false;
        }
      }

      // both are internal
      else if (rel0 >= 0 && rel1 >= 0) {
        if (rel0 != rel1 || out0 != out1) {
          return false;
        }
      }

      // one is internal, one is external
      else {
        return false;
      }
    }
  }
  return true;
}

template <typename T> std::vector<Match> getMatches(const std::vector<T *> &s) {
  RinseMatcher<T> matcher(s);
  auto matches = matcher.getRepeatedSequences();
  matches      = matcher.separateByIsomorphism(matches);
  matches      = matcher.separateByOverlaps(matches);
  matches      = matcher.getPrioritized(matches);
  matches      = matcher.getFinalMatches(matches);
  return matches;
}

} // namespace subgraph
} // namespace fwtools

#endif
