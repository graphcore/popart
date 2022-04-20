// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Overlapping0SubgraphTest

// tests of repeated sub-string matching, with no overlapping
// note overlapping means intersecting here, if which nesting is a case
//
// Note: we are testing a pure substring algorithm here,
// sub-graph isomorphisms come later. That said, in the final
// getRinseMatches function in subgraph.hpp, isomorphism partitioning
// is run before overlap partitioning. The reason for this is that
// splitting non-isomorphic matches might eliminate overlaps before
// matches are (unnecessarily) split for containing overlapping sequences

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <ostream>
#include <vector>
#include <popart/logging.hpp>

#include "blip.hpp"
#include "popart/subgraph/algo0.hpp"
#include "popart/subgraph/match.hpp"
#include "popart/subgraph/subgraphnames.hpp"
#include "popart/subgraph/subgraphutil.hpp"

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Overlapping0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 std::vector<Match> expected_matches) {
    // build up the schedule from the inputs
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(std::unique_ptr<Blip>(new Blip(t, 10.0f, {})));
    }
    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    // get matches, and compare to the expected matches
    algo0::RinseMatcherAlgo0<const Blip> rinseMatcher(sched);
    auto matches = rinseMatcher.getRepeatedSequences();
    std::sort(matches.begin(), matches.end());
    matches = separateByOverlaps(matches);
    std::sort(matches.begin(), matches.end());
    std::sort(expected_matches.begin(), expected_matches.end());

    std::stringstream ss;
    ss << "\nExpected matches:";
    for (auto &x : expected_matches) {
      ss << "\n" << x;
    }
    ss << "\nComputed matches:";
    for (auto &x : matches) {
      ss << "\n" << x;
    }
    popart::logging::debug(ss.str());
    BOOST_CHECK(matches == expected_matches);
  };

  // -------------------------------------------
  popart::logging::info("non-overlapping: 01010");
  //                            0  1  2  3  4
  std::vector<blip::Type> types{0, 1, 0, 1, 0};
  std::vector<Match> expected_matches = {
      {{0, 2, 4}, 1}, // 0
      {{1, 3}, 1},    // 1
      {{0, 2}, 2},    // 01
      {{1, 3}, 2}     // 10
                      // no 010 due to overlaps
  };
  test(types, expected_matches);

  // -------------------------------------------
  popart::logging::info("non-overlapping: 010101010");
  //                  0  1  2  3  4  5  6  7  8
  types            = {0, 1, 0, 1, 0, 1, 0, 1, 0};
  expected_matches = {
      {{0, 2, 4, 6, 8}, 1}, // 0
      {{1, 3, 5, 7}, 1},    // 1
      {{0, 2, 4, 6}, 2},    // 01
      {{1, 3, 5, 7}, 2},    // 10
      {{0, 4}, 3},          // 010, first set
      {{2, 6}, 3},          // 010, second set
      {{1, 5}, 3},          // 101
      {{0, 4}, 4},          // 0101
      {{1, 5}, 4}           // 1010
  };
  test(types, expected_matches);
}
