// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE VectorAndSetTest

#include <boost/test/unit_test.hpp>
#include <popart/vectorandset.hpp>

#include <vector>
#include <popart/logging.hpp>

BOOST_AUTO_TEST_CASE(test_vectorAndSetTest) {
  // T45403
  // As with unittest_utils.cpp it seems like we need to force linking with this
  // line
  popart::logging::debug("");

  std::vector<int> v{1, 2, 3};
  std::vector<int> v2{4, 5, 6};

  popart::VectorAndSet<int> vAS;
  popart::VectorAndSet<int> vAS2{v};
  vAS = vAS2;

  for (const auto &el : v) {
    BOOST_CHECK(vAS.contains(el));
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(
      v.begin(), v.end(), vAS2.v().begin(), vAS2.v().end());

  vAS.insert(*v.begin());
  BOOST_ASSERT(vAS.contains(*v.begin()));

  vAS.reset(v2);
  for (const auto &el : v2) {
    BOOST_CHECK(vAS.contains(el));
  }
  for (const auto &el : v) {
    BOOST_CHECK(!vAS.contains(el));
  }
}
