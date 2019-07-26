#include <algorithm>
#include <sstream>
#include <popart/subgraph/match.hpp>
#include <popart/subgraph/subgraphnames.hpp>

namespace fwtools {
namespace subgraph {

Match::Match(const std::vector<Start> &s, int l) : starts(s), length(l) {
  std::sort(starts.begin(), starts.end());
}

std::ostream &operator<<(std::ostream &stream, const Match &match) {

  stream << match.length << " | [";

  if (match.starts.size() != 0) {
    stream << match.starts[0];
  }

  if (match.starts.size() < 20) {
    for (int i = 1; i < match.starts.size(); ++i) {
      stream << ", " << match.starts[i];
    }

    stream << "]";
  }

  else {
    for (int i = 1; i < 6; ++i) {
      stream << ", " << match.starts[i];
    }
    stream << ", ... ";
    int starts_size = static_cast<int>(match.starts.size());
    for (int i = starts_size - 6; i < starts_size; ++i) {
      stream << ", " << match.starts[i];
    }

    stream << "]";
    stream << " (" << match.starts.size() << ")";
  }

  return stream;
}

// TODO (T6416) : use that it is sorted to make this algorithm linear
bool Match::crosses(const Match &rhs) const {

  if (length == 1 || rhs.length == 1) {
    return false;
  }

  for (auto s0 : starts) {
    for (auto s1 : rhs.starts) {
      auto e0 = s0 + length;
      auto e1 = s1 + rhs.length;

      //  .....sxxxe   (from s0, length 4)
      //  ........sxe  (from s1, length 2)
      if (s0 < s1 && e0 > s1 && e0 < e1) {
        return true;
      }

      // the reverse of the case above:
      else if (s1 < s0 && e1 > s0 && e1 < e0) {
        return true;
      }
    }
  }
  return false;
}

// no element of rhs is not in this Match
bool Match::contains(const Match &rhs) const {

  int thisStartIndex = 0;
  for (auto &s : rhs.starts) {
    // find the largest start which is not greater than s
    while (thisStartIndex < starts.size() - 1 &&
           starts[thisStartIndex + 1] <= s) {
      ++thisStartIndex;
    }
    if (s < starts[thisStartIndex] ||
        s + rhs.length > starts[thisStartIndex] + length) {
      return false;
    }
  }
  return true;
}

bool Match::startsIntersect(const std::vector<Start> &rhsStarts) const {
  int thisIndex = 0;
  int rhsIndex  = 0;
  while (thisIndex < starts.size() && rhsIndex < rhsStarts.size()) {
    if (starts[thisIndex] == rhsStarts[rhsIndex]) {
      return true;
    }
    if (thisIndex < rhsIndex) {
      ++thisIndex;
    } else {
      ++rhsIndex;
    }
  }
  return false;
}

bool firstContainsSecond(const std::vector<Start> &starts,
                         const std::vector<int> &rhsStarts) {

  if (starts.size() < rhsStarts.size()) {
    return false;
  }
  if (rhsStarts.size() == 0) {
    return true;
  }

  int rhs_index = 0;

  // first start greater than rhsStarts[0]
  auto nxt = std::upper_bound(starts.begin(), starts.end(), rhsStarts[0]);
  if (nxt == starts.begin()) {
    return false;
  }

  while (true) {
    if (*(nxt - 1) != rhsStarts[rhs_index]) {
      return false;
    }
    ++rhs_index;
    if (rhs_index == rhsStarts.size()) {
      return true;
    }

    // which of these 2 variants to use could depend on the vector lengths
    //
    // Variant 1 :
    // nxt = std::upper_bound(nxt, starts.end(), rhsStarts[rhs_index]);
    //
    // Variant 2 :
    while (nxt != starts.end() && *nxt <= rhsStarts[rhs_index]) {
      ++nxt;
    }
  }
}

bool Match::containsStarts(const std::vector<Start> &rhsStarts) const {
  return firstContainsSecond(starts, rhsStarts);
}

// Examples of subsuming
// (ie ++++ is subsumed)
//
//   ++++      ++++     ++++
// ******    ******   ******
// --------------------------> schedule
//
//
//  +++        +++       +++
// ******    ******   ******
// --------------------------> schedule
//
// Another example, which is NOT subsuming
// (ie ++ is NOT subsumed)
//
//   ++  ++     ++  ++    ++  ++
//   *******    *******   *******
//---------------------------------> schedule
bool Match::subsumes(const Match &rhs) const {
  if (rhs.starts.size() != starts.size()) {
    return false;
  }

  // using that starts are sorted
  for (int i = 0; i < starts.size(); ++i) {
    if (rhs.starts[i] < starts[i] ||
        rhs.starts[i] + rhs.length > starts[i] + length) {
      return false;
    }
  }
  return true;
}

// The rhs intervals map to this Match's intervals
// in a repeated way, and don't cross.  Examples:
//
// this ...xxxxx.....xxxxx
// .rhs    aa aa     aa aa     aa   aa   aa  (true)
// .rhs     aaaaaa    aaaaaa                 (false)
// .rhs    aa aa     aa aa                   (true)
// .rhs    aaaa       aaaa                   (false)

bool Match::fitsCleanly(const Match &rhs) const {

  if (rhs.length > length) {
    return false;
  }

  if (crosses(rhs)) {
    return false;
  }

  std::vector<std::vector<int>> interDeltas(starts.size());

  int currentSelfIndex = 0;
  for (int i = 0; i < rhs.starts.size(); ++i) {
    while (currentSelfIndex < starts.size() - 1 &&
           rhs.starts[i] >= starts[currentSelfIndex + 1]) {
      ++currentSelfIndex;
    }
    if (rhs.starts[i] >= starts[currentSelfIndex] &&
        rhs.starts[i] < starts[currentSelfIndex] + length) {
      interDeltas[currentSelfIndex].push_back(starts[currentSelfIndex] -
                                              rhs.starts[i]);
    }
  }

  // 1 confirm no crossings
  for (int i = 1; i < starts.size(); ++i) {
    if (interDeltas[i] != interDeltas[0]) {
      return false;
    }
  }
  return true;
}

bool Match::intersects(const Match &rhs) const {

  int thisStart = 0;
  int rhsStart  = 0;
  while (rhsStart != rhs.starts.size() && thisStart != starts.size()) {

    // a starts before b ends and a ends after b starts
    if (starts[thisStart] < rhs.starts[rhsStart] + rhs.length &&
        starts[thisStart] + starts.size() > rhs.starts[rhsStart]) {
      return true;
    }

    if (rhs.starts[rhsStart] < starts[thisStart] + length &&
        rhs.starts[rhsStart] + rhs.starts.size() > starts[thisStart]) {
      return true;
    }

    if (rhs.starts[rhsStart] < starts[thisStart]) {
      ++rhsStart;
    }

    else {
      ++thisStart;
    }
  }
  return false;
}

} // namespace subgraph
} // namespace fwtools
