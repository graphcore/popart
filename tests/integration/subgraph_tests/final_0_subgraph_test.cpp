// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Final0SubgraphTest

// tests of final greedy selection of sub-graphs

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
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

BOOST_AUTO_TEST_CASE(Final0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 const std::map<blip::Type, blip::Value> &value_map,
                 const std::vector<Match> &expected_matches,
                 float threshold = 0.0f) {
    // prepare a schedule of Blips, given the input arguments
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(t, value_map.at(t), {})));
    }

    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    // get the final matches
    algo0::RinseMatcherAlgo0<const Blip> rinseMatcher(sched);
    auto matches = rinseMatcher.getRepeatedSequences();
    matches      = separateByOverlaps(matches);

    setValues(matches, sched);
    std::sort(matches.rbegin(), matches.rend());
    matches = algo0::getFinalMatches(matches, threshold, sched.size());

    // compare the final matches to those expected in this test
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
  popart::logging::info("priorities, simple case");
  // schedule index:            0  1  2  3  4  5
  std::vector<blip::Type> types{0, 1, 2, 0, 1, 2};

  // Type 0,1 and 2 all have value 10.0f
  std::map<blip::Type, blip::Value> value_map{
      {0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  std::vector<Match> expected_matches = {
      {{0, 3}, 3} // 012 : total value = 30
  };
  test(types, value_map, expected_matches);

  // -------------------------------------------
  popart::logging::info("priorities, repetition ==> not subsumed (test 1)");
  //       0  1  2  3  4  5  6  7
  types = {0, 1, 1, 2, 0, 1, 1, 2};

  value_map = {{0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  expected_matches = {
      {{0, 4}, 4},      // 0112 : total value 40
      {{1, 2, 5, 6}, 1} // 1    :             10

  };
  test(types, value_map, expected_matches);

  // -----------------------------------------------------------------------
  popart::logging::info("priorities, repetition ==> not subsumed (test 2)");
  //       0  1  2  3  4  5  6  7
  types = {0, 1, 2, 0, 0, 1, 2, 0};

  value_map = {{0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  expected_matches = {
      {{0, 4}, 4},      // 0120 : 40
      {{0, 3, 4, 7}, 1} // 1    : 10

  };
  test(types, value_map, expected_matches);

  // -----------------------------------------------------------
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  types = {0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0};
  //       x  x  x  x  x        x  x  x  x  x
  //          ~  ~        ~  ~     ~  ~
  //               [@  @]               [@  @][@  @][@  @]
  //       +        +  +        +        +  +  +  +  +  +

  value_map = {{0, 10.0f}, {1, 11.0f}, {2, 12.0f}};

  expected_matches = {
      // first 2 matches are saturated at threshold = 0:
      {{0, 7}, 5},                              // 01200   : 50
      {{1, 5, 8}, 2},                           // 12      : 20
      {{3, 10, 12, 14}, 2},                     // 00      : 20
      {{0, 3, 4, 7, 10, 11, 12, 13, 14, 15}, 1} // 0       : 10

  };

  popart::logging::info("crossing in final means not included, threshold -1.0");
  test(types, value_map, expected_matches, -1.0f);

  expected_matches = {
      {{1, 5, 8}, 2},                           // 12      : 20
      {{0, 3, 4, 7, 10, 11, 12, 13, 14, 15}, 1} // 0       : 10
  };
  popart::logging::info("crossing in final means not included, threshold 0.0");
  test(types, value_map, expected_matches, 0.0f);

  expected_matches = {
      {{1, 5, 8}, 2},       // 12      : 20
      {{3, 10, 12, 14}, 2}, // 00      : 20
  };
  popart::logging::info("crossing in final means not included, threshold 15.0");
  test(types, value_map, expected_matches, 15.0f);
}
