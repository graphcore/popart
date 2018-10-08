#include <willow/intervals.hpp>
// for std::sqrt
#include <cmath>

namespace willow {

std::vector<std::array<int, 2>> getDecreasingIntervals(int N) {
  // what is the largest K s.t.
  // 1 + 2 + ... + K < N ?
  // the above inequality is equivalent to,
  // K(K+1) < 2*N.
  // well, K = floor(sqrt(2*N)) does not always
  //       satisfy the inequality, but K = floor(sqrt(2*N)) - 1
  //       does, so answer: floor(sqrt(2*N)) - 1.
  int K = static_cast<int>(std::sqrt(2.0 * static_cast<double>(N))) - 1;
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

} // namespace willow
