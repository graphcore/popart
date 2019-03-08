#define BOOST_TEST_MODULE Final0SubgraphTest

// tests of final greedy selection of sub-graphs

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/logging.hpp>

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Final0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 const std::map<blip::Type, blip::Value> &value_map,
                 const std::vector<Match> &expected_matches) {
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
    RinseMatcher<const Blip> rinseMatcher(sched);
    auto matches = rinseMatcher.getRepeatedSequences();
    matches      = rinseMatcher.separateByOverlaps(matches);
    matches      = rinseMatcher.getPrioritized(matches);
    matches      = rinseMatcher.getFinalMatches(matches);

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
    poponnx::logging::debug(ss.str());
    BOOST_CHECK(matches == expected_matches);
  };

  // -------------------------------------------
  poponnx::logging::info("priorities, simple case");
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
  poponnx::logging::info("priorities, repetition ==> not subsumed (test 1)");
  //       0  1  2  3  4  5  6  7
  types = {0, 1, 1, 2, 0, 1, 1, 2};

  value_map = {{0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  expected_matches = {
      {{0, 4}, 4},      // 0112 : total value 40
      {{1, 2, 5, 6}, 1} // 1    :             10

  };
  test(types, value_map, expected_matches);

  // -----------------------------------------------------------------------
  poponnx::logging::info("priorities, repetition ==> not subsumed (test 2)");
  //       0  1  2  3  4  5  6  7
  types = {0, 1, 2, 0, 0, 1, 2, 0};

  value_map = {{0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  expected_matches = {
      {{0, 4}, 4},      // 0120 : 40
      {{0, 3, 4, 7}, 1} // 1    : 10

  };
  test(types, value_map, expected_matches);

  // -----------------------------------------------------------
  poponnx::logging::info("crossing in final means not included");
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  types = {0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0};
  //       x  x  x  x  x        x  x  x  x  x
  //          ~  ~        ~  ~     ~  ~
  //               [@  @]               [@  @][@  @][@  @]
  //       +        +  +       +         +  +  +  +  +  +

  value_map = {{0, 10.0f}, {1, 10.0f}, {2, 10.0f}};

  expected_matches = {
      {{0, 7}, 5},                              // 01200   : 50
      {{3, 10, 12, 14}, 2},                     // 00      : 20
      {{1, 5, 8}, 2},                           // 12      : 20
      {{0, 3, 4, 7, 10, 11, 12, 13, 14, 15}, 1} // 1       : 10

  };
  test(types, value_map, expected_matches);
}
