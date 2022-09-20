// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/logging.hpp>
#include <popart/subgraph/algo1.hpp>
#include <popart/subgraph/match.hpp>
#include <popart/subgraph/suffixtree.hpp>

#include "popart/subgraph/subgraphnames.hpp"

namespace fwtools {
namespace subgraph {
namespace algo1 {

void Algo1Base::emplace(Match match) {

  // Check:
  // Base case where this Match is too small
  if (match.starts.size() <= 1 || match.length < 1) {
    return;
  }

  setVal(match);

  // Check:
  // Has this Match been enqueued in the past?
  if (everEnqueued.count(match) != 0) {
    return;
  }

  // Check:
  // Is there a currently enqueued Match which is has the same key (length +
  // first 5 chars) and contains all of this Match's starts.
  auto currentEnqueueKey = getCurrentEnqueueKey(match.starts[0], match.length);
  if (isDominatingEnqueued(currentEnqueueKey, match.starts)) {
    return;
  }

  // Check:
  // Is this Match subsumed by an accepted Match
  for (auto &acc : accepted) {
    if (acc.subsumes(match)) {
      return;
    }
  }

  // Checks passed, will enqueue this Match
  matchQueue.emplace(match);
  everEnqueued.emplace(match);
  auto found = currentlyEnqueued.find(currentEnqueueKey);
  if (found == currentlyEnqueued.end()) {
    currentlyEnqueued[currentEnqueueKey] = {match.starts};
  } else {
    found->second.emplace(match.starts);
  }
}

Algo1Base::Algo1Base(const std::vector<int> &intSched_, int schedSize_)
    : intSched(intSched_), schedSize(schedSize_) {

  // Add to the end of sequence a "$" to
  // make an implicit tree into an explicit tree
  intSched.push_back(-1);
}

void Algo1Base::init() {

  // we insert boundaries to reduce edge condition checks.
  edgeLocs = {};
  edgeLocs.emplace(-1000);
  edgeLocs.emplace(schedSize + 1000);

  // get internal nodes, sorted in decreasing order of length
  initMatches = fwtools::subgraph::suffixtree::getInternal(intSched);

  for (auto &m : initMatches) {
    bool subsumed = false;
    for (auto &mInQueue : everEnqueued) {
      if (mInQueue.subsumes(m)) {
        subsumed = true;
        break;
      }
    }

    // if it subsumed, it will be generated later if needed
    if (!subsumed) {
      setVal(m);
      emplace(m);
    }
  }
}

std::vector<Match> Algo1Base::getPreThresholded() {

  while (!matchQueue.empty()) {
    auto match = matchQueue.top();
    // remove Match from queue
    matchQueue.pop();

    auto currentEnqueueKey =
        getCurrentEnqueueKey(match.starts[0], match.length);

    currentlyEnqueued[currentEnqueueKey].erase(match.starts);
    if (currentlyEnqueued[currentEnqueueKey].size() == 0) {
      currentlyEnqueued.erase(currentEnqueueKey);
    }

    // Check if this Match is a valid addition to accepted,
    // generating smaller Matches where appropriate
    // if not a valid addition.
    process(match);
  }

  std::reverse(accepted.begin(), accepted.end());
  return accepted;
}

bool Algo1Base::noCrossingsWithAccepted(const Match &match) {

  // Check for crossings with already accepted matches.
  // Consider an interval of length 4:
  // .....XXXX....
  // .....^^^.....  // the region to check for crosses in
  // .....012.....
  // ...-1...3....
  //
  // s0---^
  //        ^---- s0 + length - 2

  // we check if any indexes stored in edgeLocs are any of the ^^^ indices

  if (match.length == 1) {
    // there are no ^^^ (as the number of indices to check is length - 1)
    // this is not strictly necessary, as the code below does handle this case.
    return true;
  }

  int minFirstAbove = match.length - 1;
  int maxLastBelow  = -1;
  std::vector<int> crossFreeMatches;

  for (int i = 0; i < match.starts.size(); ++i) {

    auto s0 = match.starts[i];

    // first element greater than s0-1
    auto firstAbove = *(edgeLocs.upper_bound(s0 - 1)) - s0;
    minFirstAbove   = std::min<int>(minFirstAbove, firstAbove);

    // first element greater than s0 + length - 2
    auto iter = edgeLocs.upper_bound(s0 + match.length - 2);
    // last element less than or equal to s0 + match.size - 2
    --iter;
    auto lastBelow = *iter - s0;
    maxLastBelow   = std::max<int>(maxLastBelow, lastBelow);
    if (firstAbove >= match.length - 1 && lastBelow < 0) {
      crossFreeMatches.push_back(match.starts[i]);
    }
  }

  if (crossFreeMatches.size() == match.starts.size()) {
    return true;
  }

  // Add the matches which are cross free
  // (keeping length, reducing match size)
  if (crossFreeMatches.size() > 1) {
    Match reduced_count(crossFreeMatches, match.length);
    emplace(reduced_count);
  }

  // Example:
  //  ...........xxxxxxxxxxxxxxxxx..........
  //  .............^........................
  //
  //  ...........xxxxxxxxxxxxxxxxx..........
  //  .............^^.......................
  //
  //  ...........xxxxxxxxxxxxxxxxx..........
  //  ............^.^.......................
  //
  //  ...........xxxxxxxxxxxxxxxxx..........
  //  .................^.^..................
  //                             ^-- s0 + length - 1
  //             ^------------------ s0
  //              ^----------------- minFirstAbove
  //                     ^---------- maxLastBelow
  //

  Match left_child(match.starts, maxLastBelow + 1);
  emplace(left_child);

  int right_length = match.length - 1 - minFirstAbove;
  std::vector<int> right_child_starts(match.starts);
  for (auto &x : right_child_starts) {
    x += (match.length - right_length);
  }
  Match right_child(right_child_starts, right_length);
  emplace(right_child);

  return false;
}

bool Algo1Base::allIsomorphic(const Match &match) {

  // if not all isomorphic, try left and right
  int min_delta = match.length;
  for (int i = 1; i < match.starts.size(); ++i) {
    int this_delta = isoTil(match.length, match.starts[0], match.starts[i]);
    min_delta      = std::min<int>(min_delta, this_delta);
  }

  if (min_delta == match.length) {
    return true;
  }

  if (match.length > 1) {

    // For now, adding tight left and tight right.
    Match match_left  = match;
    match_left.length = match.length - 1;
    emplace(match_left);

    auto right_starts = match.starts;
    for (auto &x : right_starts) {
      x += 1;
    }

    Match match_right(right_starts, match.length - 1);
    emplace(match_right);
  }

  // adding same lengths, partitioned into isomorphisms
  for (auto &partitioned_match : partitionedByIsomorphism(match)) {
    if (partitioned_match.starts.size() > 1) {
      emplace(partitioned_match);
    }
  }

  // also adding shifted to right if the next chars the same
  bool final_chars_equiv = true;
  for (auto &s : match.starts) {
    if ((s + match.length >= schedSize) ||
        (intSched[s + match.length] !=
         intSched[match.starts[0] + match.length])) {
      final_chars_equiv = false;
      break;
    }
  }
  if (final_chars_equiv) {
    Match shift_right = match;
    for (auto &x : shift_right.starts) {
      ++x;
    }
    emplace(shift_right);
  }

  return false;
}

Algo1Base::CurrentEnqueueKey Algo1Base::getCurrentEnqueueKey(Start s0,
                                                             int len) {
  CurrentEnqueueKey key;
  for (int i = 0; i < std::get<1>(key).size(); ++i) {
    std::get<1>(key)[i] = -7;
  }

  for (int i = 0;
       i < std::min<int>(len, static_cast<int>(std::get<1>(key).size()));
       ++i) {
    std::get<1>(key)[i] = intSched[s0 + i];
  }

  std::get<0>(key) = len;
  return key;
}

// is there an enqueued Match of length at least "l0", and
// starting indices which are a superset of "starts"

bool Algo1Base::isDominatingEnqueued(const CurrentEnqueueKey &key,
                                     const std::vector<Start> &starts) {

  auto found = currentlyEnqueued.find(key);

  if (found == currentlyEnqueued.end()) {
    return false;
  }

  // going through the enqueued vectors in order of
  // decreasing number of starts
  for (auto iter = found->second.rbegin(); iter != found->second.rend();
       ++iter) {

    if (iter->size() < starts.size()) {
      break;
    }

    if (firstContainsSecond(*iter, starts)) {
      return true;
    }
  }
  return false;
}

bool Algo1Base::noOverlapping(const Match &match) {

  // If overlapping, try emplacing smaller sub-Matches.

  // all distances between overlapping Matches:
  std::set<int> interStartDistances;
  for (int i = 1; i < match.starts.size(); ++i) {
    for (int j = i - 1; j >= 0; --j) {
      auto delta = match.starts[i] - match.starts[j];
      if (delta >= match.length) {
        break;
      }
      interStartDistances.emplace(delta);
    }
  }

  // The case where there are no overlaps
  if (interStartDistances.size() == 0) {
    return true;
  }

  int maxInterStartDistance = *interStartDistances.rbegin();

  // We will be creating non-overlapping Matches for each of the lengths
  // in interStartDistances. But first, we check for an early exit option:
  // Is there an enqueued Match which is not-shorter than
  // maxInterStartDistance, and contains all the starts of this Match?
  auto cekey = getCurrentEnqueueKey(match.starts[0], maxInterStartDistance);

  // Is an early exit
  if (isDominatingEnqueued(cekey, match.starts)) {
    interStartDistances = {};
  }

  // No early exit
  else {
  }

  // Irrespective of the early exit clause, process the case
  // of retaining the full match length
  if (match.starts.back() - match.starts[0] >= match.length) {
    interStartDistances.emplace(match.length);
  }

  // Of all the accepted matches whose intervals contain
  // all of the starts in match, which one has
  // the most starts? There may be no such accepted matches.
  Match best_accepted({}, 2);
  Match this_dummy(match);
  this_dummy.length = 1;
  for (auto &acc : accepted) {
    if (acc.starts.size() > best_accepted.starts.size() &&
        acc.contains(this_dummy)) {
      best_accepted = acc;
    }
  }

  // Before emplacing a new Match below, we will check that
  // it contains more than 2X - 1 starts than this:
  int containment_factor = static_cast<int>(best_accepted.starts.size());

  // When partitioning a Match into non-overlapping Matches,
  // we greedily add starts to the first new Match it doesn't overlap with.
  // This generates just 1 partitioning (of exponentially many). To generate
  // more partitions, we consider starting the process with start indices:
  // [0, ... nFirstSetters)
  int nFirstSetters =
      std::min<int>(3, static_cast<int>(match.starts.size()) - 1);

  for (auto sub_length : interStartDistances) {

    // for each firstSetter, what do the matches
    // (up to index nFirstSetters) look like? We use
    // a std::set, to remove duplicates
    std::set<std::vector<Match>> initial_locals;

    std::vector<Start> initialOtherStarts = std::vector<int>(
        match.starts.begin() + 1, match.starts.begin() + nFirstSetters);

    // as we proceed through this loop, initialOtherStarts has match starts,
    // @iteration 0   [1,2,3,....n-1,n]
    // @iteration 1   [0,2,3,....n-1,n]
    // @iteration 2   [0,1,3,....n-1,n]
    // .
    // .
    // @iteration n-1 [0,1,2,....n-2,n]
    // @iteration n   [0,1,2,....n-2, n-1]
    for (int firstSetter = 0; firstSetter < nFirstSetters; ++firstSetter) {
      if (firstSetter > 0) {
        initialOtherStarts[firstSetter - 1] = match.starts[firstSetter - 1];
      }
      std::vector<Match> local = {{{match.starts[firstSetter]}, sub_length}};
      completeLocalUnsorted(local, initialOtherStarts, sub_length);
      initial_locals.emplace(local);
    }

    std::vector<int> otherStarts = std::vector<int>(
        match.starts.begin() + nFirstSetters, match.starts.end());

    for (auto &match_set : initial_locals) {
      std::vector<Match> sub_matches;
      for (auto m : match_set) {
        sub_matches.push_back(m);
      }
      completeLocalSorted(sub_matches, otherStarts, sub_length);

      for (auto m : sub_matches) {
        if (m.starts.size() >= 2 * std::max(1, containment_factor)) {
          emplace(m);
        } else {
          // not emplacing, as it is dominated by best_accepted
        }
      }
    }
  }

  return false;
}

void Algo1Base::completeLocalSorted(std::vector<Match> &local,
                                    const std::vector<Start> &otherStarts,
                                    int sub_length) const {
  for (auto &s : otherStarts) {
    bool matchFound = false;
    for (auto &m : local) {
      bool withinDistance = (s - m.starts.back()) < sub_length;
      if (!withinDistance) {
        matchFound = true;
        m.starts.push_back(s);
        // break from loop over local
        break;
      }
    }
    if (matchFound == false) {
      local.push_back({{s}, sub_length});
    }
  }
}

void Algo1Base::completeLocalUnsorted(std::vector<Match> &local,
                                      const std::vector<Start> &otherStarts,
                                      int sub_length) const {

  for (auto &s : otherStarts) {
    bool matchFound = false;
    for (auto &m : local) {
      bool withinDistance = false;
      for (auto s0 : m.starts) {
        int d0 = std::abs(s - s0);
        if (d0 < sub_length) {
          withinDistance = true;
          // break from loop over m.starts
          break;
        }
      }
      if (!withinDistance) {
        matchFound = true;
        m.starts.push_back(s);
        // break from loop over local
        break;
      }
    }
    if (matchFound == false) {
      local.push_back({{s}, sub_length});
    }
  }

  for (auto &m : local) {
    std::sort(m.starts.begin(), m.starts.end());
    // local.push_back(m);
  }
}

// how to process the top of the priority queue
void Algo1Base::process(const Match &match) {
  if (match.length < 1 || match.starts.size() < 2) {
    // base case
    return;
  }

  if (match.getDiscountedValue() < 0.0) {
    // match has negative value
    popart::logging::trace(
        "[RINSE Algo1] Negative discounted value: {} (value: {}, length: {})",
        match.getDiscountedValue(),
        match.getValue(),
        match.length);
    // Child matches can have a higher subgraph value than the parent match
    auto left_starts  = match.starts;
    auto right_starts = match.starts;
    for (size_t i = 0; i < right_starts.size(); ++i) {
      ++right_starts[i];
    }
    Match left_child(left_starts, match.length - 1);
    Match right_child(right_starts, match.length - 1);
    emplace(left_child);
    emplace(right_child);
    return;
  }

  for (auto &acc : accepted) {

    if (acc.length == match.length && acc.startsIntersect(match.starts)) {
      // same length as accepted with intersecting starts
      popart::logging::trace("[RINSE Algo1] Intersecting starts, same length.");
      return;
    }

    // if "subsumed", discard this match,
    // don't generate smaller cases as
    // they will also be subsumed
    if (acc.subsumes(match)) {
      popart::logging::trace("[RINSE Algo1] Subsumed.");
      return;
    }

    if (acc.contains(match) && (2 * acc.starts.size() > match.starts.size())) {
      // contained by accepted with at least 1/2X the starts
      popart::logging::trace("[RINSE Algo1] Contained (1/2x starts).");
      return;
    }
  }

  if (!noCrossingsWithAccepted(match)) {
    popart::logging::trace("[RINSE Algo1] Crossing.");
    return;
  }

  if (!noOverlapping(match)) {
    popart::logging::trace("[RINSE Algo1] Overlapping.");
    return;
  }

  if (!allIsomorphic(match)) {
    popart::logging::trace("[RINSE Algo1] Not isomorphic.");
    return;
  }

  for (auto &acc : accepted) {
    if (acc.intersects(match)) {
      popart::logging::trace("[RINSE Algo1] Intersects.");
      // the only type of intersection which can be
      // accepted is a "clean fit" (see definition)
      if (!acc.fitsCleanly(match)) {
        // contained but not cleanly
        popart::logging::trace("[RINSE Algo1] Not contained cleanly.");
        return;
      }
    }
  }

  popart::logging::trace(
      "[RINSE Algo1] Accepting [{}, [{}]]",
      match.length,
      popart::logging::join(match.starts.begin(), match.starts.end(), ", "));
  accepted.push_back(match);

  if (match.length > 1) {
    for (auto &s0 : match.starts) {
      edgeLocs.emplace(s0 - 1);
      edgeLocs.emplace(s0 + match.length - 1);
    }
  }
}

std::vector<int>
getSequenceBreaks(const std::vector<std::pair<size_t, size_t>> &sequences_) {
  std::vector<int> breaks(sequences_.size() + 1);

  for (auto &seq : sequences_) {
    if (seq.second - seq.first > 1) {
      ++breaks.at(seq.first);
      ++breaks.at(seq.second);
    }
  }

  for (size_t i = 1; i < breaks.size(); ++i) {
    breaks[i] += breaks[i - 1];
  }

  return breaks;
}

} // namespace algo1
} // namespace subgraph
} // namespace fwtools
