// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UnittestUtils

#include <sstream>

#include <boost/test/unit_test.hpp>

#include <popart/logging.hpp>
#include <popart/util.hpp>

namespace {
/**
 * Helper function to trigger a call to the stream operator.
 */
template <typename T> std::string toString(const T &t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}
} // namespace

BOOST_AUTO_TEST_CASE(unittest_utils_streamoperator_pair) {
  // T41277. This test was failing with `libpva.so: undefined symbol: dlsym`.
  // Investigation showed that this test was not linking libpopart.so. Adding
  // this logging call does not fix the underlying issue, but does force this
  // test to link against popart, allowing us to continue running the tests.
  popart::logging::debug("");

  // Test operator<< for std::pair.
  BOOST_TEST("(5, test_value)" == toString(std::make_pair(5, "test_value")));
  BOOST_TEST("(-1, j)" == toString(std::make_pair(-1, 'j')));
}

BOOST_AUTO_TEST_CASE(unittest_utils_streamoperator_tuple) {
  // Test operator<< for std::tuples.
  BOOST_TEST("(5, test_value)" == toString(std::make_tuple(5, "test_value")));
  BOOST_TEST("(-1, j, 3)" == toString(std::make_tuple(-1, 'j', 3)));
}
