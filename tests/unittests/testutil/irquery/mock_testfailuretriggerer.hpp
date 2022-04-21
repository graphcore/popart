// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_IR_QUERY_MOCK_TEST_FAILURE_TRIGGERER_HPP
#define TEST_UTILS_IR_QUERY_MOCK_TEST_FAILURE_TRIGGERER_HPP

#include <boost/test/unit_test.hpp>
#include <boost/trompeloeil.hpp>

#include <testutil/irquery/testfailuretriggerer.hpp>

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

#endif
