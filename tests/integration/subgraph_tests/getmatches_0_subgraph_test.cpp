// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Final0SubgraphTest

// tests of final greedy selection of sub-graphs

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <popart/logging.hpp>
#include <popart/subgraph/outliner.hpp>

#include "blip.hpp"
#include "popart/subgraph/match.hpp"
#include "popart/subgraph/subgraphnames.hpp"

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Final0_Subgraph) {

  auto test = [](const std::vector<blip::Type> &types,
                 const std::map<blip::Type, blip::Value> &value_map,
                 const std::vector<Edge> &edges,
                 std::vector<Match> expected_matches,
                 int algo,
                 float threshold) {
    // prepare the schedule from the input parameters
    std::vector<const Blip *> sched;
    std::vector<std::unique_ptr<Blip>> blips;
    for (auto &t : types) {
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(t, value_map.at(t), {})));
    }

    for (auto edge : edges) {
      blips[edge.destId]->addIn(
          edge.inIndex, blips[edge.sourceId].get(), edge.outIndex);

      blips[edge.sourceId]->addOut(blips[edge.destId].get(), edge.outIndex);
    }
    for (int i = 0; i < blips.size(); ++i) {
      sched.push_back(blips[i].get());
    }

    std::vector<std::pair<size_t, size_t>> sequences(sched.size());
    float sequenceBreakCost = 0.0f;

    // get the matches
    std::vector<Match> matches;
    if (algo == 1) {
      matches = getRinseMatches<const Blip>(sched,
                                            sequences,
                                            threshold,
                                            sequenceBreakCost,
                                            OutlinerAlgorithm::ALGO1);
    } else if (algo == 0) {
      matches = getRinseMatches<const Blip>(sched,
                                            sequences,
                                            threshold,
                                            sequenceBreakCost,
                                            OutlinerAlgorithm::ALGO0);
    } else {
      throw std::runtime_error("invalid algo");
    }

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
    ss << "\n\n\n";
    popart::logging::debug(ss.str());
    BOOST_CHECK(matches == expected_matches);
  };

  // -------------------------------------------
  //                            0  1  2  3  4  5  6  7  8  9
  std::vector<blip::Type> types{0, 1, 1, 0, 1, 1, 0, 1, 1, 0};
  //                            x  x  x  x        x  x  x  x
  //                               w  w     w  w     w  w
  //                              [*][*]   [*][*]   [*][*]
  //                            ^        ^        ^        ^
  // 1's are worth slightly more that 0's
  std::map<blip::Type, blip::Value> value_map{{0, 10.0f}, {1, 11.0f}};

  std::vector<Match> expected_matches = {
      {{0, 6}, 4},             // 0110
      {{1, 4, 7}, 2},          // 11
      {{1, 2, 4, 5, 7, 8}, 1}, // 1
      {{0, 3, 6, 9}, 1}        // 0

  };
  std::vector<Edge> edges;

  for (int algo : {0, 1}) {
    popart::logging::info(
        "full test, saturated large sequence, threshold -1.0, algo {}", algo);
    test(types, value_map, edges, expected_matches, algo, -1.0);
  }

  expected_matches = {
      {{1, 2, 4, 5, 7, 8}, 1}, // 1
      {{0, 3, 6, 9}, 1}        // 0
  };

  for (int algo : {0, 1}) {
    popart::logging::info(
        "full test, saturated large sequence, threshold 5.0, algo {}", algo);
    test(types, value_map, edges, expected_matches, algo, 5.0);
  }

  // -------------------------------------------
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  types = {6, 1, 2, 5, 1, 2, 6, 6, 1, 2, 7, 1, 2, 8, 1, 2};
  //       0--0
  //       0-----0
  //       0-----1
  //       0-----------------------0
  //       0--------------------------0
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
           {0, 2, 0, 1}, // 0
           {0, 8, 0, 0},
           {0, 9, 0, 0},
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

  auto metatest = [types, value_map, edges, &test](
                      float threshold, std::vector<Match> expected_matches) {
    for (int algo : {0, 1}) {
      test(types, value_map, edges, expected_matches, algo, threshold);
    }
  };

  expected_matches = {
      // 12 first isomorphic type
      // The other 2 isomorphic types (starting at 1 and 8) are singletons
      // Note, that with a non-negative threshold, this match is removed
      {{4, 11, 14}, 2},
      // 1: it is not subsumed
      {{4, 8, 11, 14}, 1},
      // 2: it is not subsumed. The 2 at index 2 has the same input tensor
      {{5, 9, 12, 15}, 1},
      // 6: index 0 has it's output consumed, so not isomorphic here
      {{6, 7}, 1}};

  popart::logging::info(
      "full test, non-isomorphic double take, threshold -1.0f");
  metatest(-1.0f, expected_matches);

  popart::logging::info("metatest, threshold 0.0f");
  metatest(0.0f, {{{4, 8, 11, 14}, 1}, {{5, 9, 12, 15}, 1}, {{6, 7}, 1}});

  popart::logging::info("metatest, threshold 9.0f");
  metatest(9.0f, {{{4, 8, 11, 14}, 1}, {{5, 9, 12, 15}, 1}});

  popart::logging::info("metatest, threshold 100.0f");
  metatest(100.0f, {});
}
