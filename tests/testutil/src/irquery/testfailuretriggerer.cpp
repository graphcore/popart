// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <iostream>
#include <string>
#include <testutil/irquery/testfailuretriggerer.hpp>

namespace popart {
namespace irquery {

void TestFailureTriggerer::trigger(const std::string &errorMsg) {
  // Previously, this test used BOOST_REQUIRE_MESSAGE, however the inclusion of
  // Boost:unit_test_framework was leading to double frees when running tests.
  // To avoid this, we simply terminate.
  throw std::runtime_error(errorMsg);
}
} // namespace irquery
} // namespace popart
