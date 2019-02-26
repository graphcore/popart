#define BOOST_TEST_MODULE Region0Test

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/region.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(Region_Scale0) {
  view::Region r0({0, 0}, {3, 3});
  view::Region r1({3, 3}, {6, 6});
  BOOST_CHECK(r0.intersect(r1).isEmpty());
  BOOST_CHECK(r1.intersect(r0).isEmpty());
}
