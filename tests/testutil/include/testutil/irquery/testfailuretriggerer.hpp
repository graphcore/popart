// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef TEST_UTILS_TEST_FAILURE_TRIGGERER_HPP
#define TEST_UTILS_TEST_FAILURE_TRIGGERER_HPP

#include <string>

namespace popart {
namespace irquery {

/**
 * Abstract base class for something that can trigger a test failure.
 */
class TestFailureTriggerer {
public:
  /**
   * Function that triggers a test failure with a message. It's basically a
   * wrapper around BOOST_REQUIRE that is exposed so that IrQuery objects
   * themselves can be tested.
   **/
  virtual void trigger(const std::string &errorMsg);
};

} // namespace irquery
} // namespace popart

#endif
