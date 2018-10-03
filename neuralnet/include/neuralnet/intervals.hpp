#ifndef GUARD_NEURALNET_INTERVALS_HPP
#define GUARD_NEURALNET_INTERVALS_HPP

#include<vector>
#include <array>

namespace neuralnet{

// split [0,N) into intervals, s.t. each subsequent 
// interval has width 1 less than the previous one
// (as far as possible).
std::vector<std::array<int, 2>> getDecreasingIntervals(int N);

}


#endif

