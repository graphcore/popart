// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SampleTest

#include <boost/test/unit_test.hpp>
#include <poplar/Tensor.hpp>

BOOST_AUTO_TEST_CASE(Sample) {
  // Checks everything is set up correctly.
  poplar::Tensor t;
  // TODO remove this test as soon as we have a real test case.
  BOOST_CHECK(true);
}
