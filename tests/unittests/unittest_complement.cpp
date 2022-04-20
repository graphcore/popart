// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Complement
#include <boost/test/unit_test.hpp>
#include <initializer_list>
#include <popart/commgroup.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/collectives.hpp>

#include "popart/error.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(Complement) {
  popart::Ir ir;
  auto get_complement_all = getComplementCommGroup;
  auto get_complement     = getComplementCommGroupWithSuperSet;
  auto same               = [](CommGroup a, CommGroup b) {
    return a.type == b.type && a.replicaGroupSize == b.replicaGroupSize;
  };

  auto sizes = {0, 2, 4, 6, 8};

  auto cgts = {CommGroupType::All,
               CommGroupType::None,
               CommGroupType::Consecutive,
               CommGroupType::Orthogonal};

  auto all = CommGroup{CommGroupType::All, 0};

  auto none = get_complement(ir, all, all);

  BOOST_REQUIRE(none.type == CommGroupType::None);

  for (auto &type : cgts) {
    for (auto &size : sizes) {
      if ((type == CommGroupType::All || type == CommGroupType::None) &&
          size != 0) {
        continue;
      }
      if ((type == CommGroupType::Consecutive ||
           type == CommGroupType::Orthogonal) &&
          size == 0) {
        continue;
      }

      CommGroup superSet = {type, (unsigned int)size};

      // crosschecks
      for (auto &type_ : cgts) {
        for (auto &size_ : sizes) {
          if ((type_ == CommGroupType::All || type_ == CommGroupType::None) &&
              size_ != 0) {
            continue;
          }
          if ((type_ == CommGroupType::Consecutive ||
               type_ == CommGroupType::Orthogonal) &&
              size_ == 0) {
            continue;
          }

          CommGroup commgroup = {type_, (unsigned int)size_};

          auto expect_fail =
              !(same(superSet, all) || same(superSet, commgroup));

          auto forwarded = same(superSet, all);

          try {
            auto complement = get_complement(ir, commgroup, superSet);

            if (forwarded) {
              BOOST_REQUIRE(
                  same(complement, get_complement_all(ir, commgroup)));
            } else {
              BOOST_REQUIRE(same(complement, none));
            }
          } catch (const internal_error &ie) {
            if (!expect_fail) {
              throw;
            }
            BOOST_REQUIRE(expect_fail);
          }
        }
      }
    }
  }
}
