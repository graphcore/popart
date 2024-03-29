// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_ALGO0_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_ALGO0_HPP_

#include <vector>

#include "popart/subgraph/match.hpp"
#include "popart/subgraph/subgraphutil.hpp"
#include "rinsematcherbase.hpp"

namespace fwtools {
namespace subgraph {
namespace algo0 {

std::vector<Match> getRepeatedIntSequences(const std::vector<int> &intSched);

// template paramter T might be popart::Op, or
// const tf::Op (is that the tensorflow class?)
template <typename T> class RinseMatcherAlgo0 : public RinseMatcherBase<T> {
public:
  RinseMatcherAlgo0(const std::vector<T *> &s) : RinseMatcherBase<T>(s) {}
  // sub-string matching for repeated sequences
  std::vector<Match> getRepeatedSequences() const {
    return getRepeatedIntSequences(
        getIntSchedule(RinseMatcherBase<T>::schedule));
  }
};

// get the final Matches from the prioritised vector of Matches,
// not taking Matches which are subsumed by, or those that cross over
// already accepted Matches. Also applies the threshold.
std::vector<Match> getFinalMatches(const std::vector<Match> &matches,
                                   float threshold,
                                   int schedSize);

} // namespace algo0
} // namespace subgraph
} // namespace fwtools

#endif // POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_ALGO0_HPP_
