// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExceptionTest

#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/error.hpp>

#include "popart/logging.hpp"

void test_function1() { throw popart::error("Test {} {} {}", 1, 2, 3); }

BOOST_AUTO_TEST_CASE(LoggingTest1) {

  try {
    test_function1();
    BOOST_TEST(false);
  } catch (const popart::error &e) {
    BOOST_CHECK(e.what() == std::string("Test 1 2 3"));
  }
}

void test_function2() { throw popart::error("Test {} {} {}", 1, 2); }

BOOST_AUTO_TEST_CASE(LoggingTest2) {

  try {
    test_function2();
    BOOST_TEST(false);
  } catch (const popart::error &e) {
    BOOST_CHECK(
        e.what() ==
        std::string("Popart exception format error argument not found"));
  }
}
