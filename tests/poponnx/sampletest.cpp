#define BOOST_TEST_MODULE SampleTest

#include <poplar/Tensor.hpp>
#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <boost/test/unit_test.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

BOOST_AUTO_TEST_CASE(Sample) {
  // Checks everything is set up correctly.
  poplar::Tensor t;
  // TODO remove this test as soon as we have a real test case.
  BOOST_CHECK(true);
}
