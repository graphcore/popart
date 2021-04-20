// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <boost/test/unit_test.hpp>

#include <testutil/irquery/testfailuretriggerer.hpp>

namespace popart {
namespace irquery {

void TestFailureTriggerer::trigger(const std::string &errorMsg) {
  BOOST_REQUIRE_MESSAGE(false, errorMsg);
}

} // namespace irquery
} // namespace popart
