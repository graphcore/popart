#define BOOST_TEST_MODULE Final0SubgraphTest

// tests of final greedy selection of sub-graphs

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <poponnx/logging.hpp>

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Final0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 const std::map<blip::Type, blip::Value> &value_map,
                 const std::vector<Edge> &edges,
                 std::vector<Match> expected_matches) {
    // prepare the schedule from the input parameters
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(t, value_map.at(t), {})));
    }
    for (auto edge : edges) {
      blips[edge.destId]->ins[edge.inIndex] = {
          blips[edge.sourceId].get(), edge.outIndex, ""};
    }
    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    // get the matches
    auto matches = getMatches<const Blip>(sched);

    // compare to the expected matches
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
  poponnx::logging::info("full test, saturated large sequence");
  //                            0  1  2  3  4  5  6  7  8  9
  std::vector<blip::Type> types{0, 1, 1, 0, 1, 1, 0, 1, 1, 0};
  //                            x  x  x  x        x  x  x  x
  //                               w  w     w  w     w  w
  //                              [*][*]   [*][*]   [*][*]
  //                            ^        ^        ^        ^
  // TODO: T7255 check for sub-sequences which are saturated by smaller
  // sequences, as in the case above. Match "w" should be removed

  // 1's are worth slightly more that 0's
  std::map<blip::Type, blip::Value> value_map{{0, 10.0f}, {1, 11.0f}};

  std::vector<Match> expected_matches = {
      {{0, 6}, 4},             // 0110
      {{1, 4, 7}, 2},          // 11
      {{1, 2, 4, 5, 7, 8}, 1}, // 1
      {{0, 3, 6, 9}, 1}        // 0

  };
  std::vector<Edge> edges;
  test(types, value_map, edges, expected_matches);

  // -------------------------------------------
  poponnx::logging::info("full test, non-isomorphic double take");
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  types = {6, 1, 2, 5, 1, 2, 6, 6, 1, 2, 7, 1, 2, 8, 1, 2};
  //       0--0
  //       0----0
  //          0--1
  //          0--------------------0
  //          0-----------------------0
  //                               0--1
  //       0-----------0
  //       0--------------1  note here this is 0->1, not 0->0
  //                   0--0  as in the first 2. This third edge changed too
  //       0--------------------------------0
  //       0-----------------------------------1
  //                                        0--0
  //
  //       0------------------------------------------0
  //       0---------------------------------------------1
  //                                                  0--0

  edges = {{0, 1, 0, 0}, // 0
           {0, 2, 0, 0}, // 0
           {1, 2, 0, 1}, // 0
           {1, 8, 0, 0},
           {1, 9, 0, 0},
           {8, 9, 0, 1},
           {0, 4, 0, 0}, // 0
           {0, 5, 0, 1}, // 1
           {4, 5, 0, 0}, // 0 i.e. different input indices => not isomorphism
           {0, 11, 0, 0},
           {0, 12, 0, 1},
           {11, 12, 0, 0},
           {0, 14, 0, 0},
           {0, 15, 0, 1},
           {14, 15, 0, 0}

  };

  // 1's are worth slightly more that 0's
  value_map = {{1, 11.0f},
               {2, 10.0f},
               {3, 1.0f},
               {4, 1.0f},
               {5, 1.0f},
               {6, 1.0f},
               {7, 1.0f},
               {8, 1.0f}};

  expected_matches = {
      {{4, 11, 14}, 2},       // 12 first isomorphic type
      {{1, 8}, 2},            // 12 second isomorphic type
      {{1, 4, 8, 11, 14}, 1}, // 1: it is not subsumed
      {{2, 5, 9, 12, 15}, 1}, // 2: it is not subsumed
      {{0, 6, 7}, 1}          // 6
  };
  test(types, value_map, edges, expected_matches);
}
