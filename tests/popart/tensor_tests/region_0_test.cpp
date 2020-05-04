// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Region0Test

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/region.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Region_Scale0) {
  view::Region r0({0, 0}, {3, 3});
  view::Region r1({3, 3}, {6, 6});
  BOOST_CHECK(r0.intersect(r1).isEmpty());
  BOOST_CHECK(r1.intersect(r0).isEmpty());
}

BOOST_AUTO_TEST_CASE(Region_Sub0) {
  view::Region r0({0, 0}, {4, 4});
  view::Region r1({1, 1}, {3, 3});

  view::Regions subs = r0.sub(r1);

  BOOST_CHECK(subs.size() == 4);
  BOOST_CHECK(subs[0].intersect(r1).isEmpty());
  BOOST_CHECK(subs[1].intersect(r1).isEmpty());
  BOOST_CHECK(subs[2].intersect(r1).isEmpty());
  BOOST_CHECK(subs[3].intersect(r1).isEmpty());
}

BOOST_AUTO_TEST_CASE(Region_Sub1) {
  view::Region r0({0, 0}, {6, 6});
  view::Region r1({0, 0}, {4, 4});
  view::Region r2({3, 3}, {7, 7});
  view::Region r3({0, 4}, {3, 6});
  view::Region r4({4, 0}, {6, 3});

  view::Regions subs = r0.sub({r1, r2});

  BOOST_CHECK(subs.size() == 2);

  BOOST_CHECK(subs.front() == r3 || subs.back() == r3);
  BOOST_CHECK(subs.front() == r4 || subs.back() == r4);
  BOOST_CHECK(subs.front() != subs.back());
}

BOOST_AUTO_TEST_CASE(Region_Merge0) {
  view::Region r0({3, 3}, {8, 8});
  view::Region r1({0, 3}, {3, 8});

  auto merge = r0.merge(r1);

  BOOST_CHECK(!merge.second.isEmpty());
  BOOST_CHECK(merge.first == 0);
  BOOST_CHECK(merge.second == view::Region({0, 3}, {8, 8}));
}

BOOST_AUTO_TEST_CASE(Region_Reshape0) {
  view::Region r0({0, 0}, {6, 6});
  view::Region r1({0, 0, 0, 0}, {2, 3, 2, 3});
  view::Region r2({2, 2}, {4, 5});

  view::Regions r = r2.reshape(r0, r1);

  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({0, 2, 0, 2}, {1, 3, 1, 3})) != r.end());
  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({0, 2, 1, 0}, {1, 3, 2, 2})) != r.end());
  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({1, 0, 0, 2}, {2, 1, 1, 3})) != r.end());
  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({1, 0, 1, 0}, {2, 1, 2, 2})) != r.end());
}

BOOST_AUTO_TEST_CASE(Region_Reshape1) {
  view::Region r0({0, 0}, {8, 8});
  view::Region r1({0, 0, 0}, {8, 2, 4});
  view::Region r2({2, 1}, {7, 6});

  view::Regions r = r2.reshape(r0, r1);

  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({2, 1, 0}, {7, 2, 2})) != r.end());
  BOOST_CHECK(std::find(r.begin(),
                        r.end(),
                        view::Region({2, 0, 1}, {7, 1, 4})) != r.end());
}

BOOST_AUTO_TEST_CASE(Region_Reshape2) {
  view::Region r0({0, 0}, {1024, 2});
  view::Region r1({0, 0, 0}, {1, 1024, 2});

  view::Regions r = r0.reshape(r0, r1);

  BOOST_CHECK(std::find(r.begin(), r.end(), r1) != r.end());
}

BOOST_AUTO_TEST_CASE(Region_Reshape3) {
  view::Region r0({0, 0}, {2, 80});
  view::Region r1({0, 0, 0, 0}, {2, 5, 4, 4});

  view::Regions r2 = r0.reshape(r0, r1);
  view::Regions r3 = r1.reshape(r1, r0);

  BOOST_CHECK(r2.size() == 1 && r2.front() == r1);
  BOOST_CHECK(r3.size() == 1 && r3.front() == r0);
}

BOOST_AUTO_TEST_CASE(Region_Reshape4) {
  view::Region r0({0}, {160});
  view::Region r1({0, 0, 0, 0}, {2, 5, 4, 4});

  view::Regions r2 = r0.reshape(r0, r1);
  view::Regions r3 = r1.reshape(r1, r0);

  BOOST_CHECK(r2.size() == 1 && r2.front() == r1);
  BOOST_CHECK(r3.size() == 1 && r3.front() == r0);
}

BOOST_AUTO_TEST_CASE(Region_Reshape5) {
  view::Region r0({0, 16}, {2, 64});
  view::Region r1({0, 0}, {2, 80});
  view::Region r2({0, 0, 0, 0}, {2, 5, 4, 4});
  view::Region r3({0, 1, 0, 0}, {2, 4, 4, 4});

  view::Regions r4 = r0.reshape(r1, r2);
  view::Regions r5 = r3.reshape(r2, r1);

  BOOST_CHECK(r4.size() == 1 && r4.front() == r3);
  BOOST_CHECK(r5.size() == 1 && r5.front() == r0);
}

BOOST_AUTO_TEST_CASE(Region_Reshape6) {
  view::Region r0({0, 2}, {2, 5});
  view::Region r1({0, 0}, {2, 8});
  view::Region r2({0}, {16});
  view::Region r3({2}, {5});
  view::Region r4({10}, {13});

  view::Regions r5 = r0.reshape(r1, r2);
  view::Regions r6 = view::mergeRegions(
      {r5.front().reshape(r2, r1).front(), r5.back().reshape(r2, r1).front()});

  BOOST_CHECK(r5.size() == 2 && r5.front() == r3 && r5.back() == r4);
  BOOST_CHECK(r6.size() == 1 && r6.front() == r0);
}

BOOST_AUTO_TEST_CASE(Region_MergeRegions0) {

  view::Region r0({0, 0}, {3, 4}, view::AccessType::Read);
  view::Region r1({1, 3}, {5, 5}, view::AccessType::Write);

  view::Regions rs = view::mergeRegions({r0, r1});

  BOOST_CHECK(rs.size() == 3);

  auto r2 = std::find(rs.begin(), rs.end(), view::Region({0, 0}, {3, 4}));
  BOOST_CHECK(r2 != rs.end());
  BOOST_CHECK(r2->getAccessType() == view::AccessType::ReadWrite);

  auto r3 = std::find(rs.begin(), rs.end(), view::Region({3, 3}, {5, 4}));
  BOOST_CHECK(r3 != rs.end());
  BOOST_CHECK(r3->getAccessType() == view::AccessType::ReadWrite);

  auto r4 = std::find(rs.begin(), rs.end(), view::Region({1, 4}, {5, 5}));
  BOOST_CHECK(r4 != rs.end());
  BOOST_CHECK(r4->getAccessType() == view::AccessType::ReadWrite);
}

BOOST_AUTO_TEST_CASE(Region_AccessType0) {

  BOOST_CHECK(view::combine({view::AccessType::None, view::AccessType::Read}) ==
              view::AccessType::Read);
  BOOST_CHECK(
      view::combine({view::AccessType::None, view::AccessType::Write}) ==
      view::AccessType::Write);
  BOOST_CHECK(
      view::combine({view::AccessType::None, view::AccessType::ReadWrite}) ==
      view::AccessType::ReadWrite);
  BOOST_CHECK(
      view::combine({view::AccessType::Read, view::AccessType::Write}) ==
      view::AccessType::ReadWrite);
  BOOST_CHECK(
      view::combine({view::AccessType::Read, view::AccessType::ReadWrite}) ==
      view::AccessType::ReadWrite);
  BOOST_CHECK(
      view::combine({view::AccessType::Write, view::AccessType::ReadWrite}) ==
      view::AccessType::ReadWrite);
}
