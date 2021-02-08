// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RandomBaseTest

#include <popart/op/randombase.hpp>

#include <boost/test/unit_test.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(RandomSeedPlaceholder_test) {

  RandomSeedPlaceholder p1;
  RandomSeedPlaceholder p2;
  RandomSeedPlaceholder p3;

  // Distinctly created placeholders are not equal.
  BOOST_CHECK(!(p1 == p2));
  BOOST_CHECK(!(p1 == p3));
  BOOST_CHECK(!(p2 == p3));

  // Copies-constructed placeholders are equal to what they copied.
  RandomSeedPlaceholder p1Copy(p1);
  BOOST_CHECK(p1Copy == p1);
  BOOST_CHECK(!(p1Copy == p2));

  // Assigned placeholders are equal after assignment.
  p3 = p1;
  BOOST_CHECK(p3 == p1);
  BOOST_CHECK(!(p3 == p2));
  p2 = p3;
  BOOST_CHECK(p3 == p2);
  BOOST_CHECK(p3 == p1);
}