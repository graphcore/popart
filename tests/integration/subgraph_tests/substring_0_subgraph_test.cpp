// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Substring0SubgraphTest

// tests of repeated sub-string matching. This is the
// first step in the sub-graph matching algorithm, and
// is a pure string matching algorithm

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/logging.hpp>

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Substring0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 std::vector<Match> expected_matches) {
    // prepare the schedule based off of the input arguments
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(std::unique_ptr<Blip>(new Blip(t, 10.0f, {})));
    }
    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    // perform the repeated sub-string matching step
    algo0::RinseMatcherAlgo0<const Blip> rinseMatcher(sched);
    auto matches = rinseMatcher.getRepeatedSequences();
    std::sort(matches.begin(), matches.end());
    std::sort(expected_matches.begin(), expected_matches.end());

    // and compare to the expected matches
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

  // first test
  // ----------
  std::vector<blip::Type> types{0, 1, 0, 1, 0, 1, 2, 3};
  std::vector<Match> expected_matches;

  // length 1 starts:
  // {0,2,4}
  // {1,3,5}
  expected_matches.push_back({{0, 2, 4}, 1});
  expected_matches.push_back({{1, 3, 5}, 1});

  // length 2 starts:
  // {0,2,4}
  // {1,3}
  expected_matches.push_back({{0, 2, 4}, 2});
  expected_matches.push_back({{1, 3}, 2});

  // length 3 starts:
  // {0,2}
  // {1,3}
  expected_matches.push_back({{0, 2}, 3});
  expected_matches.push_back({{1, 3}, 3});

  // length 4 starts:
  // {0,2}
  expected_matches.push_back({{0, 2}, 4});

  test(types, expected_matches);

  // second test
  // -----------
  //                  0  1  2  3  4  5  6  7  8  9
  types            = {0, 1, 2, 3, 0, 1, 2, 0, 1, 0};
  expected_matches = {
      {{0, 4, 7, 9}, 1}, // 0
      {{1, 5, 8}, 1},    // 1
      {{2, 6}, 1},       // 2
      {{0, 4, 7}, 2},    // 01
      {{1, 5}, 2},       // 12
      {{0, 4}, 3}        // 012
  };
  test(types, expected_matches);

  // third test
  // ----------
  //                  0  1  2  3  4  5  6  7  8  9  10 11
  types            = {0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1};
  expected_matches = {
      {{0, 1, 2, 6, 7, 10}, 1}, // 0
      {{3, 4, 5, 8, 9, 11}, 1}, // 1
      {{0, 1, 6}, 2},           // 00
      {{3, 4, 8}, 2},           // 11
      {{2, 7, 10}, 2},          // 01
      {{5, 9}, 2},              // 10
      {{1, 6}, 3},              // 001
      {{2, 7}, 3},              // 011
      {{4, 8}, 3}               // 110
  };

  // fourth test
  // -----------
  //       0  1  2  3  4  5    6    7  8  9  10 11  12   13   14  15  16  17
  types = {1, 2, 3, 4, 5, 100, 100, 6, 7, 8, 9, 10, 100, 100, 11, 12, 13, 1};
  expected_matches = {
      {{5, 6, 12, 13}, 1}, //"100"
      {{0, 17}, 1},        // "1"
      {{5, 12}, 2}         // "100" "100"
  };
  test(types, expected_matches);
}
