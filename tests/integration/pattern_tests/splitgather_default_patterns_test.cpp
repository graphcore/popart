// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SplitGatherTestDefault

#include <boost/test/unit_test.hpp>
#include <popart/patterns/patterns.hpp>

#include "popart/logging.hpp"

BOOST_AUTO_TEST_CASE(SplitGatherTest1) {

  using namespace popart;

  // check that SPLITGATHER is only on for level ALL

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isSplitGatherEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isSplitGatherEnabled() == false);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isSplitGatherEnabled() == true);
}
