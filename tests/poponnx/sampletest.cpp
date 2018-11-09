#define BOOST_TEST_MODULE SampleTest

#include <poplar/Tensor.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Sample) {
  // Checks everything is set up correctly.
  poplar::Tensor t;
  // TODO remove this test as soon as we have a real test case.
  BOOST_CHECK(true);
}
