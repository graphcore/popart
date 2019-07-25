#define BOOST_TEST_MODULE SplitGatherTestDefault

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/patterns/patterns.hpp>

BOOST_AUTO_TEST_CASE(SplitGatherTest1) {

  using namespace popart;

  // check that SPLITGATHER is only on for level ALL

  Patterns noPatterns(PatternsLevel::NONE);
  BOOST_CHECK(noPatterns.isSplitGatherEnabled() == false);

  Patterns defPatterns(PatternsLevel::DEFAULT);
  BOOST_CHECK(defPatterns.isSplitGatherEnabled() == false);

  Patterns allPatterns(PatternsLevel::ALL);
  BOOST_CHECK(allPatterns.isSplitGatherEnabled() == true);
}
