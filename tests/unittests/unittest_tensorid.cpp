// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TensorIdTest

#include <boost/test/unit_test.hpp>

#include <popart/logging.hpp>
#include <popart/util.hpp>

BOOST_AUTO_TEST_CASE(test_getBaseTensorId) {
  auto test = [](const char *input, const char *expected) {
    auto baseId = popart::getBaseTensorId(input);
    popart::logging::debug("checking '{}' == '{}'", baseId, expected);
    BOOST_CHECK_EQUAL(baseId, expected);
  };

  // Some general use cases
  test("sometensor__t1", "sometensor");
  test("some__t1__tensor__t2", "some__t1__tensor");

  // Check some partial matches
  test("100", "100");
  test("t0", "t0");
  test("_t1", "_t1");
  test("__t1", "__t1");
  test("__t", "__t");
}
