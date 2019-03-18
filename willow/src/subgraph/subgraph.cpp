#include "poponnx/subgraph/subgraph.hpp"

namespace fwtools {
namespace subgraph {

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

bool areCrossing(int seq_length0,
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

} // namespace subgraph
} // namespace fwtools
