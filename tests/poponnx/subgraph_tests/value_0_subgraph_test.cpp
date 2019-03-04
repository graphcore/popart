#define BOOST_TEST_MODULE Value0SubgraphTest

// tests of value ordering of the matches

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/logging.hpp>

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Value0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 const std::map<blip::Type, blip::Value> &value_map,
                 std::vector<Match> expected_matches) {
    // construct a schedule from the input values to this test
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(t, value_map.at(t), {})));
    }
    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    RinseMatcher<const Blip> rinseMatcher(sched);
    auto matches = rinseMatcher.getRepeatedSequences();
    matches      = rinseMatcher.getPrioritized(matches);

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
  //                            0  1  2  3  4  5
  std::vector<blip::Type> types{0, 1, 2, 0, 1, 2};

  std::map<blip::Type, blip::Value> value_map{
      {0, 5.0f}, {1, 11.0f}, {2, 13.0f}};

  std::vector<Match> expected_matches = {
      {{0, 3}, 3}, // 012 : 29
      {{1, 4}, 2}, // 12  : 24
      {{0, 3}, 2}, // 01  : 16
      {{2, 5}, 1}, // 2   : 13
      {{1, 4}, 1}, // 1   : 11
      {{0, 3}, 1}  // 0   : 5
  };
  test(types, value_map, expected_matches);

  // -------------------------------------------
  poponnx::logging::info("priorities, length rules");
  //       0  1  2  3  4  5  6  7  8  9
  types = {0, 1, 2, 3, 0, 1, 0, 1, 2, 3};

  value_map = {{0, 9.0f}, {1, 1.0f}, {2, 4.0f}, {3, 6.0f}};

  expected_matches = {
      {{0, 6}, 4},    // 0123 : 20.0
      {{0, 6}, 3},    // 012  : 14.0
      {{1, 7}, 3},    // 123  : 11.0
      {{0, 4, 6}, 2}, // 01   : 10.0  as there are 3 sequences here.
      {{2, 8}, 2},    // 23   : 10.0  and only 2 here, 3 takes precedence.
      {{0, 4, 6}, 1}, // 0    : 9.0
      {{3, 9}, 1},    // 3    : 6.0
      {{1, 7}, 2},    // 12   : 5.0
      {{2, 8}, 1},    // 2    : 4.0
      {{1, 5, 7}, 1}  // 1    : 1.0
  };
  test(types, value_map, expected_matches);
}
