// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_INTERVALS_HPP
#define GUARD_NEURALNET_INTERVALS_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace popart {

// split [0,N) into intervals, s.t. each subsequent
// interval has width 1 less than the previous one
// (as far as possible). Think: solution to the problem
// of minimising the maximum number of egg drops to determine
// highest safe floor of a building to drop an egg from, given
// 2 eggs.
std::vector<std::array<int, 2>> getDecreasingIntervals(int N);

std::vector<std::array<int, 2>>
getDecreasingIntervals(const std::vector<int64_t> &floorHeights);

} // namespace popart

#endif
