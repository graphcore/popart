#define BOOST_TEST_MODULE MatchBaseTests

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/logging.hpp>
#include <popart/subgraph/match.hpp>

using namespace fwtools::subgraph;

BOOST_AUTO_TEST_CASE(Match_intersect) {

  // x.x.x....
  // ....x....
  Match m0({0, 2, 4}, 1);
  Match m1({4}, 1);
  BOOST_CHECK(m0.intersects(m1));

  // x.x.x....
  // .....x...
  m0 = {{0, 2, 4}, 1};
  m1 = {{5}, 1};
  BOOST_CHECK(!m0.intersects(m1));

  // xxx.......xxx...
  // .....xxxx.......
  m0 = {{0, 10}, 3};
  m1 = {{5}, 4};
  BOOST_CHECK(!m0.intersects(m1));

  // xxx.......xxx...
  // .......xxxx.....
  m0 = {{0, 10}, 3};
  m1 = {{7}, 4};
  BOOST_CHECK(m0.intersects(m1));

  // x....x.x.x..x
  // ...xx.....xx.xxxx
  m0 = {{0, 5, 7, 9, 12}, 1};
  m1 = {{3, 10, 13, 15}, 2};
  BOOST_CHECK(!m0.intersects(m1));

  // x....x.x.x...x
  // ...xx.....xx.xxxx
  m0 = {{0, 5, 7, 9, 13}, 1};
  m1 = {{3, 10, 13, 15}, 2};
  BOOST_CHECK(m0.intersects(m1));

  m0 = {{1000}, 2};
  m1 = {{1000}, 3};
  BOOST_CHECK(m0.intersects(m1));

  m0 = {{1000}, 2};
  m1 = {{1001}, 3};
  BOOST_CHECK(m0.intersects(m1));

  m0 = {{1000}, 2};
  m1 = {{1002}, 3};
  BOOST_CHECK(!m0.intersects(m1));
}
