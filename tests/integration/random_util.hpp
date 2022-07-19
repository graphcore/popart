// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_INTEGRATION_RANDOM_UTIL_HPP_
#define POPART_TESTS_INTEGRATION_RANDOM_UTIL_HPP_

#include <algorithm>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>

namespace popart {

// The default pseudo-random number engine to use
using DefaultRandomEngine = std::mt19937;

// Boost Random ensures numerical consistency across implementations.
template <typename T>
using UniformRealDistribution = boost::random::uniform_real_distribution<T>;

// Boost Random ensures numerical consistency across implementations.
template <typename T>
using UniformIntDistribution = boost::random::uniform_int_distribution<T>;

// Create a random string with the given length
std::string randomString(size_t length) {
  std::random_device dev;
  auto seed = dev();
  DefaultRandomEngine eng(seed);
  UniformIntDistribution<uint64_t> idis(0,
                                        std::numeric_limits<uint64_t>::max());

  auto randchar = [&idis, &eng]() -> char {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[idis(eng) % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

} // namespace popart

#endif // POPART_TESTS_INTEGRATION_RANDOM_UTIL_HPP_
