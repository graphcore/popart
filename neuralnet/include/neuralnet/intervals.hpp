#ifndef GUARD_NEURALNET_INTERVALS_HPP
#define GUARD_NEURALNET_INTERVALS_HPP

#include <array>
#include <vector>

namespace neuralnet {

// split [0,N) into intervals, s.t. each subsequent
// interval has width 1 less than the previous one
// (as far as possible). Think: solution to the problem 
// of minimising the maximum number of egg drops to determine
// highest safe floor of a building to drop an egg from, given 
// 2 eggs.
std::vector<std::array<int, 2>> getDecreasingIntervals(int N);

} // namespace neuralnet

#endif
