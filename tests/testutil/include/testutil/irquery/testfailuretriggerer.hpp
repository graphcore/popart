// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTFAILURETRIGGERER_HPP_
#define POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTFAILURETRIGGERER_HPP_

#include <string>

namespace popart {
namespace irquery {

/**
 * Abstract base class for something that can trigger a test failure.
 */
class TestFailureTriggerer {
public:
  virtual ~TestFailureTriggerer() = default;
  /**
   * Function that triggers a test failure with a message. It's basically a
   * wrapper around BOOST_REQUIRE that is exposed so that IrQuery objects
   * themselves can be tested.
   **/
  virtual void trigger(const std::string &errorMsg);
};

} // namespace irquery
} // namespace popart

#endif // POPART_TESTS_TESTUTIL_INCLUDE_TESTUTIL_IRQUERY_TESTFAILURETRIGGERER_HPP_
