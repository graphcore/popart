// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_MOCK_TESTFAILURETRIGGERER_HPP_
#define POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_MOCK_TESTFAILURETRIGGERER_HPP_

#include <boost/test/unit_test.hpp> // IWYU pragma: keep
#include <boost/trompeloeil.hpp>
#include <string>

#include "testutil/irquery/irquery.hpp"

namespace popart {
namespace irquery {

/**
 * We provide a mock implementation of TestFailureTriggerer because when testing
 * some TestWrapper implementations we want to check if BOOST_REQUIRE is called
 * at the right times.
 */
class MockTestFailureTriggerer : public TestFailureTriggerer {
public:
  MAKE_MOCK1(trigger, void(const std::string &), override);
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_UNITTESTS_TESTUTIL_IRQUERY_MOCK_TESTFAILURETRIGGERER_HPP_
