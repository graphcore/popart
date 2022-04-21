// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RINSEMATCHER_ALGO1_HPP
#define GUARD_NEURALNET_RINSEMATCHER_ALGO1_HPP

#include "match.hpp"
#include "rinsematcherbase.hpp"

#include <array>
#include <queue>

namespace fwtools {
namespace subgraph {
namespace algo1 {

class VectorCompare {
public:
  bool operator()(const std::vector<Start> &a,
                  const std::vector<Start> &b) const {
    if (a.size() != b.size()) {
      return a.size() < b.size();
    }
    return a < b;
  }
};

class Algo1Base {

public:
  Algo1Base(const std::vector<int> &intSched_, int schedSize);
  virtual ~Algo1Base() = default;
  std::vector<Match> getPreThresholded();
  void init();

private:
  // all accepted Matches (before thresholding)
  std::vector<Match> accepted;

  //   suppose we're considering adding
  //   ....XXXXX.....XXXXX......
  //
  //   to accepted. If there are starts (^)
  //   at any of the following locations,
  //
  //   ....XXXXX.....XXXXX......
  //   .....^^^^......^^^^......
  //
  //   or ends (^) at any of these locations,
  //
  //   ....XXXXX.....XXXXX......
  //   ....^^^^......^^^^.......
  //
  //   then there will be a crossing.
  //   This is guaranteed as the matches are
  //   considered in an order such that subsequent
  //   intervals do not subsume earlier ones.
  //
  //   Examples:
  //   .......XXXXXX
  //   ........[........] => is a crossing
  //   .......[.......]   => not a crossing
  //   ....[.......]      => not a crossing
  //   ..[........]       => is a crossing
  //
  //   When a match is accepted, its starts and
  //   ends must be inserted into the set below.
  //   More precisely, start - 1 and start + length -1
  //   will be inserted into the set edgeLocs.
  //   Then, locations in [start, start+length-1) of XXXXX
  //   are checked for in edgeLocs.
  //
  std::set<int> edgeLocs;

  // initialize the priority queue as the internal nodes of the suffix
  // tree. We are guaranteed that every possible match is contained in
  // one of these (as a prefix/suffix of one of them).
  std::vector<Match> initMatches;

  std::priority_queue<Match> matchQueue;

  // all Matches ever put onto matchQueue
  std::set<Match> everEnqueued;

  // length and first N ints in subsequence
  using CurrentEnqueueKey = std::tuple<int, std::array<int, 8>>;

  CurrentEnqueueKey getCurrentEnqueueKey(Start s0, int len);

  // Key : (1) length, (2) first 5 elements of the
  // integer sequence of the match (or -7s if length is less than 5)
  // Value : Starts of Match
  // Value sorted by 1) number of starts, 2) starts
  std::map<CurrentEnqueueKey, std::set<std::vector<Start>, VectorCompare>>
      currentlyEnqueued;

  // convert the schedule into an integer schedule for the suffix-tree
  std::vector<int> intSched;

  int schedSize;

  bool isDominatingEnqueued(const CurrentEnqueueKey &,
                            const std::vector<Start> &);

  void process(const Match &match);
  bool noCrossingsWithAccepted(const Match &match);
  bool allIsomorphic(const Match &match);
  bool noOverlapping(const Match &match);
  void emplace(Match match);

  void completeLocalSorted(std::vector<Match> &local,
                           const std::vector<Start> &otherStarts,
                           int sub_length) const;

  void completeLocalUnsorted(std::vector<Match> &local,
                             const std::vector<Start> &otherStarts,
                             int sub_length) const;

  virtual void setVal(Match &)                                       = 0;
  virtual void setVals(std::vector<Match> &)                         = 0;
  virtual int isoTil(int, Start, Start)                              = 0;
  virtual std::vector<Match> partitionedByIsomorphism(const Match &) = 0;
};

std::vector<int>
getSequenceBreaks(const std::vector<std::pair<size_t, size_t>> &sequences_);

template <typename T> class Algo1 : public Algo1Base {
public:
  Algo1(const std::vector<T *> &sched,
        const std::vector<std::pair<size_t, size_t>> &sequences_,
        float sequenceBreakCost_)
      : Algo1Base(getIntSchedule(sched), static_cast<int>(sched.size())),
        rmb(sched), cumVals(getCumVals(sched)), sequences(sequences_),
        sequenceBreaks(getSequenceBreaks(sequences_)),
        sequenceBreakCost(sequenceBreakCost_) {}

private:
  // Using cumulative to accelerate to make this O(1)
  void setVal(Match &match) final {
    // Initial value of the match
    double val =
        cumVals[match.starts[0] + match.length] - cumVals[match.starts[0]];
    match.setValue(val);

    // Decrease match value if the match breaks sequences
    for (Start start : match.starts) {

      // Number of sequence breaks in this match
      auto numBreaks = sequenceBreaks.at(start + match.length - 1) -
                       sequenceBreaks.at(start);

      // Sequence broken, reduce value of the match
      val -= static_cast<float>(numBreaks) * sequenceBreakCost /
             static_cast<float>(match.starts.size());

      // Test if the match encases a schedule position (sequence.at(i))
      // that commands a sequence
      // Examples of cases that will be tested:
      //   Sequence: ..XSXX... (where S at position i)
      //   Match:    .XXXXX... (match encases S)
      //   Sequence: ..XSXX... (where S at position i)
      //   Match:    ..XX..... (match encases S)
      // Case that will not be tested:
      //   Sequence: ..XSXX... (where S at position i)
      //   Match:    ....XX... (match does not encase S)
      for (Start i = start; i < start + match.length; ++i) {
        if (start <= sequences.at(i).first &&
            start + match.length >= sequences.at(i).second) {
          // Match subsumes whole sequence, boost value to correct for
          // sequence breaks that have been subtracted unnecessarily
          // Sequence: ..XSXX... (where S at position i)
          // Match:    .XXXXXX..
          float factor = 0.f;
          if (start < sequences.at(i).first) {
            // Not a perfect match before sequence
            // Sequence: ..XSXX... (where S at position i)
            // Match:    .XXXXX...
            //            ^
            ++factor;
          }
          if (start + match.length > sequences.at(i).second) {
            // Not a perfect match after sequence
            // Sequence: ..XSXX... (where S at position i)
            // Match:    ..XXXXX..
            //                 ^
            ++factor;
          }
          if (sequences.at(i).second - sequences.at(i).first > 1) {
            val += factor * sequenceBreakCost /
                   static_cast<float>(match.starts.size());
          }
        }
        if (start >= sequences.at(i).first &&
            start + match.length <= sequences.at(i).second &&
            match.length < sequences.at(i).second - sequences.at(i).first) {
          // Sequence subsumes whole match, but the match is not spanning the
          // whole sequence
          // Match encases position i, which should be outlined in sequence,
          // but the match only overlaps with a part of the sequence
          // Sequence: ..XSXX... (where S at position i)
          // Match:    ..XX.....
          val -= sequenceBreakCost / static_cast<float>(match.starts.size());
        }
      }
    }
    match.setDiscountedValue(val);
  }

  // This function should use the cumulative in the same way as setVal
  void setVals(std::vector<Match> &matches) final {
    setValues<T>(matches, rmb.schedule);
  }
  int isoTil(int len, Start s0, Start s1) final {
    return isomorphicUntil<T>(len, s0, s1, rmb.schedule, rmb.schedule_index);
  }

  std::vector<Match> partitionedByIsomorphism(const Match &match) final {
    return rmb.separateSingleMatchByIsomorphism(match);
  }

  RinseMatcherBase<T> rmb;

  // Cumulative sum of subgraph values
  std::vector<double> cumVals;

  // Sequences that add a penalty when broken
  const std::vector<std::pair<size_t, size_t>> sequences;

  // Cumulative sum of number of sequence breaks
  std::vector<int> sequenceBreaks;

  // Penalty for broken sequences
  float sequenceBreakCost;
};

} // namespace algo1
} // namespace subgraph
} // namespace fwtools

#endif
