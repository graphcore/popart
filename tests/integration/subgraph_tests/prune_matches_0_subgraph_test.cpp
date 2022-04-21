// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PruneMatches0SubgraphTest

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <popart/logging.hpp>
#include <popart/subgraph/prunematches.hpp>

using namespace fwtools::subgraph;
using namespace blip;

namespace {

template <class CostModel>
void test(const std::vector<blip::Type> &types,
          const std::vector<Edge> &edges,
          std::vector<Match> expected_matches_before,
          std::vector<Match> expected_matches_after,
          float threshold = -1.0f) {
  // prepare the schedule from the inputs
  std::vector<const Blip *> sched;
  std::vector<std::unique_ptr<Blip>> blips;
  for (auto &t : types) {
    blips.emplace_back(std::unique_ptr<Blip>(new Blip(t, 10.0f, {})));
  }
  for (auto edge : edges) {
    blips[edge.destId]->addIn(
        edge.inIndex, blips[edge.sourceId].get(), edge.outIndex);
    blips[edge.sourceId]->addOut(blips[edge.destId].get(), edge.outIndex);
  }
  for (int i = 0; i < blips.size(); ++i) {
    sched.push_back(blips[i].get());
  }

  // get the matches
  algo0::RinseMatcherAlgo0<const Blip> rinseMatcher(sched);
  auto matches = rinseMatcher.getRepeatedSequences();
  matches      = separateByOverlaps(matches);
  setValues(matches, sched);
  std::sort(matches.rbegin(), matches.rend());
  // no thresholding in initial match generation
  matches = algo0::getFinalMatches(matches, -1, sched.size());

  std::sort(matches.begin(), matches.end());
  std::sort(expected_matches_before.begin(), expected_matches_before.end());

  std::stringstream ss;
  ss << "\nExpected matches before prune:";
  for (auto &x : expected_matches_before) {
    ss << "\n" << x;
  }
  ss << "\nComputed matches before prune:";
  for (auto &x : matches) {
    ss << "\n" << x;
  }
  popart::logging::debug(ss.str());
  BOOST_CHECK(matches == expected_matches_before);

  matches = fwtools::subgraph::prune::pruneMatches<const Blip, CostModel>(
      matches, sched, threshold);

  std::sort(matches.begin(), matches.end());
  std::sort(expected_matches_after.begin(), expected_matches_after.end());

  std::stringstream ss2;
  ss2 << "\nExpected matches after prune:";
  for (auto &x : expected_matches_after) {
    ss2 << "\n" << x;
  }
  ss2 << "\nComputed matches after prune:";
  for (auto &x : matches) {
    ss2 << "\n" << x;
  }
  popart::logging::debug(ss2.str());
  BOOST_CHECK(matches == expected_matches_after);
}
} // namespace

BOOST_AUTO_TEST_CASE(PruneMatches) {

  // ----------------------------------------------------------
  popart::logging::info("isomorphism preserving edges test");

  std::vector<blip::Type> types;
  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //                0-----0           (output index 0 -> input index 0)
  //                         0-----0
  //                1--2              (output index 1 -> input index 2)
  //                         1--2
  //
  //       x  x  x  XXXXXXX  XXXXXXX

  std::vector<blip::Edge> edges;
  edges = {{3, 5, 0, 0}, {6, 8, 0, 0}, {3, 4, 1, 2}, {6, 7, 1, 2}};

  std::vector<Match> expected_matches_before;
  expected_matches_before = {
      {{0, 1, 2}, 1}, // 0
      {{3, 6}, 3}     // 1 2 3
  };
  // Under the Mod 3 cost model, the final length 3 cost is pruned to length 2
  std::vector<Match> expected_matches_after = {
      {{0, 1, 2}, 1}, //
      {{4, 7}, 2}     //
  };
  test<ModThreeCostModel>(
      types, edges, expected_matches_before, expected_matches_after);

  types                   = {0, 1, 2, 0, 1, 2, 5, 5, 5, 5, 5, 5};
  edges                   = {};
  expected_matches_before = {
      {{6, 7, 8, 9, 10, 11}, 1}, //
      {{0, 3}, 3},               //
      {{6, 9}, 3}                //
  };

  expected_matches_after = {
      // {{6, 7, 8, 9, 10, 11}, 1}, //
      {{1, 4}, 2}, //
      {{7, 10}, 2} //
  };
  test<ModThreeCostModel>(
      types, edges, expected_matches_before, expected_matches_after, 15.0f);
}

BOOST_AUTO_TEST_CASE(PruneMatches_Regression1) {

  class CustomCostModel {
  public:
    float value(int64_t begin,
                int64_t end,
                const std::vector<const Blip *> &sched,
                const std::map<const Blip *, int> &sched_index) {

      // A cost model that favours our error case.
      if (begin > 0) {
        // pick longest
        return (end - begin) * 1e-3;
      } else {
        // take priority over other starts and pick shortest, unless it's really
        // short
        if (end - begin >= 2)
          return 1 + (end - begin) * -1e-3;
        else
          return 0;
      }
    }
  };

  // ----------------------------------------------------------
  popart::logging::info("testing pruneMatches regression1");

  std::vector<blip::Type> types;
  // clang-format off
  //
  //            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
  types      = {0, 1, 2, 3, 5, 6, 0, 1, 2, 3, 7, 1, 2, 3, 8, 1, 2};

  // Expected matches:
  // m_1           *  *              *  *        *  *        *  *
  // m_2           *  *  *           *  *  *     *  *  *
  // m_3        *  *  *  *        *  *  *  *
  //
  // Expected state when pruning m_3 (after processing m_1, m_2 as whole):
  // isInternal 0  0  1  0  0  0  0  0  1  0  0  0  1  0  0  0  0
  // isFirst    0  1  0  0  0  0  0  1  0  0  0  1  0  0  0  1  0
  // isLast     0  0  1  1  0  0  0  0  1  1  0  0  1  1  0  0  1
  //
  // Resulting in:
  // goodStart  1  1  0  0  1  1  1  1  0  0  1  1  0  0  1  1  0
  // goodEnd       1  0  0  1  1  1  1  0  0  1  1  0  0  1  1  0
  //
  // And hence the following considered prunings are:
  // m_3'       *                 *
  // m_3'       *  *  *  *        *  *  *  *
  // m_3'          *  *  *           *  *  *
  //
  // Previous implementation considered this (erroneous) pruning:
  // m_3'       *  *  *           *  *  *
  //
  // clang-format on

  std::vector<blip::Edge> edges;
  edges = {};

  std::vector<Match> expected_matches_before;
  expected_matches_before = {
      {{1, 7, 11, 15}, 2}, // m_1
      {{1, 7, 11}, 3},     // m_2
      {{0, 6}, 4},         // m_3
  };

  // Under the Mod 3 cost model, the final length 3 cost is pruned to length 2
  std::vector<Match> expected_matches_after = {
      {{1, 7, 11, 15}, 2}, // m_1
      {{1, 7, 11}, 3},     // m_2
      {{0, 6}, 4},         // m_3
  };
  test<CustomCostModel>(
      types, edges, expected_matches_before, expected_matches_after);
}