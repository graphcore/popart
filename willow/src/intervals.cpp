// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/intervals.hpp>
// for std::sqrt
#include <algorithm>
#include <cmath>
#include <numeric>

namespace popart {

std::vector<std::array<int, 2>> getDecreasingIntervals(int N) {
  // what is the largest K s.t.
  // 1 + 2 + ... + K < N ?
  // the above inequality is equivalent to,
  // K(K+1) < 2*N.
  // well, K = floor(sqrt(2*N)) does not always
  //       satisfy the inequality, but K = floor(sqrt(2*N)) - 1
  //       does, so lower bound on answer: floor(sqrt(2*N)) - 1.
  int K = static_cast<int>(std::sqrt(2.0 * static_cast<double>(N))) - 1;

  // by computing a bit less, experiments with resnet-50 show marginal (<1%)
  // increase in memory with big (>10%) cycle saving, this should be
  // investigated as part of T9345
  K = static_cast<int>(static_cast<float>(K) * 0.75f);

  if (N == 1) {
    return {{{0, 1}}};
  } else {
    std::vector<std::array<int, 2>> intervals;
    int i0           = 0;
    int intervalSize = K;

    while (i0 + intervalSize <= N && intervalSize > 0) {
      intervals.push_back({{i0, i0 + intervalSize}});
      i0 = i0 + intervalSize;
      --intervalSize;
    }
    while (i0 < N) {
      intervals.push_back({{i0, i0 + 1}});
      ++i0;
    }
    return intervals;
  }
}

// Examples for the (above) equally spaced floors function:
// N = 1:
// [0,1)
//
// N = 2: (K = 1)
// [0,1) [1,2)
//
// N = 3: (K = 1)
// [0,1) [1,2) [2,3)
//
// N = 4: (K = 1)
// [0,1) [1,2) [2,3) [3,4)
//
// N = 5: (K = 2)
// [0,2) [2,3) [3,4) [4,5)
//
// N = 6: (K = 2)
// [0,2) [2,3) [3,4) [4,5) [5,6)
//
// N = 7: (K = 2)
// [0,2) [2,3) [3,4) [4,5) [5,6) [6,7)
//
// N = 8: (K = 3)
// [0,3) [3,5) [5,6) [6,7) [7,8)
//
// .
// .
// .
//
// N = 25: (K = 6)
// [0,6) [6,11) [11,15) [15,18) [18, 20) [20, 21) [21, 22) ...

// Currently this more precise algorithm produces larger memory footprints
// (for resnet-50). This, and other algorithms, should be investigated
// see https://phabricator.sourcevertex.net/T9345
std::vector<std::array<int, 2>>
getDecreasingIntervals(const std::vector<int64_t> &floorHeights) {

  int N = static_cast<int>(floorHeights.size());

  // K < N intervals of decreasing size, with final interval [N-1, N).
  auto equalSpacingSoln = getDecreasingIntervals(N);

  std::vector<int64_t> accuHeights(N + 1, 0);

  // sum_{} sum_{0} sum_{0,1} ... sum_{0,1...N-1}
  std::partial_sum(
      floorHeights.begin(), floorHeights.end(), accuHeights.begin() + 1);

  auto totalHeight = accuHeights.back();

  // normalize so that final entry is N
  std::vector<int> oldToNew;
  oldToNew.reserve(N + 1);
  for (auto x : accuHeights) {
    oldToNew.push_back(static_cast<int>((N * x) / totalHeight));
  }
  // Example, oldToNew might look like this for N = 5
  // @0 0 (always 0)
  // @1 2
  // @2 4
  // @3 4
  // @4 5
  // @5 5 (always N)
  //
  std::vector<int> revOldToNew(N + 1, -1);
  for (int i = 0; i < N + 1; ++i) {
    auto revIndex = oldToNew[i];
    if (revOldToNew[revIndex] == -1) {
      revOldToNew[revIndex] = i;
    }
  }
  // At this point revOldToNew looks like:
  // @0 0
  // @1 -1
  // @2 1
  // @3 -1
  // @4 2
  // @5 4
  // we fill in the -1s
  for (int i = 1; i < N + 1; ++i) {
    if (revOldToNew[i] == -1) {
      revOldToNew[i] = revOldToNew[i - 1];
    }
  }
  // To get:
  // @0 0
  // @1 0
  // @2 1
  // @3 2
  // @4 2
  // @5 4

  std::vector<std::array<int, 2>> adjustedIntervals;

  for (auto interval : equalSpacingSoln) {
    auto oldStart = std::get<0>(interval);
    auto oldEnd   = std::get<1>(interval);
    auto newStart = revOldToNew[oldStart];
    auto newEnd   = std::max(newStart + 1, revOldToNew[oldEnd]);
    adjustedIntervals.push_back({{newStart, newEnd}});
  }
  return adjustedIntervals;
}

} // namespace popart
