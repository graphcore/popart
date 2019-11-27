#ifndef GUARD_NEURALNET_RINSEMATCHER_ALGO1_HPP
#define GUARD_NEURALNET_RINSEMATCHER_ALGO1_HPP

#include "match.hpp"
#include "rinsematcherbase.hpp"
#include "suffixtree.hpp"

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

  bool isDominatingEnqueued(const CurrentEnqueueKey &,
                            const std::vector<Start> &);

  int schedSize;

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

template <typename T> class Algo1 : public Algo1Base {
public:
  Algo1(const std::vector<T *> &sched)
      : Algo1Base(getIntSchedule(sched), static_cast<int>(sched.size())),
        rmb(sched), cumVals(getCumVals(sched)) {}

private:
  // Using cumulative to accelerate to make this O(1)
  void setVal(Match &match) final {
    auto val =
        cumVals[match.starts[0] + match.length] - cumVals[match.starts[0]];
    match.setValue(val);
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
  std::vector<float> cumVals;
};

} // namespace algo1
} // namespace subgraph
} // namespace fwtools

#endif
