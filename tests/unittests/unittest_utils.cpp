// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UnittestUtils

#include <sstream>
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <popart/logging.hpp>
#include <popart/scope.hpp>
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

BOOST_AUTO_TEST_CASE(unittest_utils_split_string) {
  // Tests that splitting of string is working.
  std::string test = "Never give in - never, never, never, never, in nothing "
                     "great or small, large or petty, never give in except to "
                     "convictions of honour and good sense";
  std::string delimiter = "never";
  std::vector<std::string> expected{
      "Never give in - ",
      ", ",
      ", ",
      ", ",
      ", in nothing great or small, large or petty, ",
      " give in except to convictions of honour and good sense"};

  auto result = popart::splitString(test, delimiter);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      result.begin(), result.end(), expected.begin(), expected.end());

  test      = "Without delimiter";
  delimiter = "/";
  expected  = {test};
  result    = popart::splitString(test, delimiter);
  BOOST_CHECK_EQUAL_COLLECTIONS(
      result.begin(), result.end(), expected.begin(), expected.end());
}

BOOST_AUTO_TEST_CASE(unittest_add_and_remove_scopes) {
  popart::TensorId tId = "g1/g2/g3/name";

  popart::Scope s4;
  s4 = s4 / "g4";

  auto result = addScope(s4, tId);

  popart::TensorId expected = "g4/g1/g2/g3/name";
  BOOST_CHECK_EQUAL(result, expected);
}
