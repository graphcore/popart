// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UnittestUtils

#include <boost/algorithm/string/predicate.hpp>
#include <boost/test/unit_test.hpp>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/util.hpp>

#include "popart/error.hpp"
#include "popart/tensordebuginfo.hpp"

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

bool checkErrorMsgRemoveScopes(const popart::error &ex) {
  const auto expectedPrefix = "Cannot remove scope from ";
  return boost::algorithm::starts_with(ex.what(), expectedPrefix);
}

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

BOOST_AUTO_TEST_CASE(unittest_add_and_remove_scopes) {
  popart::TensorId tId = "g1/g2/g3/name";

  popart::Ir ir;
  auto &g1 = ir.createGraph({"g1"});
  auto &g2 = ir.createGraph({"g2"});
  auto &g3 = ir.createGraph({"g3"});
  auto &g4 = ir.createGraph({"g4"});

  BOOST_CHECK_EXCEPTION(
      removeScope(g2, tId), popart::error, checkErrorMsgRemoveScopes);
  auto result = removeScope(g1, tId);
  result      = removeScope(g2, result);
  result      = removeScope(g3, result);
  result      = addScope(g4, result);

  popart::TensorId expected = "g4/name";
  BOOST_CHECK_EQUAL(result, expected);
}
