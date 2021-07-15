// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Isomorphism0SubgraphTest

// tests for isomorphism (I)

#include "blip.hpp"
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>
#include <popart/logging.hpp>

BOOST_AUTO_TEST_CASE(Isomorphism0_Subgraph) {

  using namespace fwtools::subgraph;
  using namespace blip;

  auto test = [](const std::vector<blip::Type> &types,
                 const std::vector<Edge> &edges,
                 std::vector<Match> expected_matches) {
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
    auto matches0 = rinseMatcher.getRepeatedSequences();
    auto matches1 = rinseMatcher.separateMultipleMatchesByIsomorphism(matches0);

    // as this test has does not include the match priority functionality
    // we sort the matches (and the expected matches) to make comparing
    // them easier
    std::sort(matches1.begin(), matches1.end());
    std::sort(expected_matches.begin(), expected_matches.end());

    std::stringstream ss;
    ss << "\nExpected matches:";
    for (auto &x : expected_matches) {
      ss << "\n" << x;
    }
    ss << "\nComputed matches:";
    for (auto &x : matches1) {
      ss << "\n" << x;
    }
    popart::logging::debug(ss.str());
    BOOST_CHECK(matches1 == expected_matches);
  };

  // -----------------------------------------------
  popart::logging::info("no edges test");

  //                               0  1  2  3  4  5  6  7  8  9
  std::vector<blip::Type> types = {0, 1, 2, 3, 0, 1, 2, 0, 1, 0};

  std::vector<Match> expected_matches = {
      {{0, 4, 7, 9}, 1}, // 0
      {{1, 5, 8}, 1},    // 1
      {{2, 6}, 1},       // 2
      {{0, 4, 7}, 2},    // 01
      {{1, 5}, 2},       // 12
      {{0, 4}, 3}        // 012
  };
  std::vector<Edge> edges;
  test(types, edges, expected_matches);

  // ----------------------------------------------------------
  popart::logging::info("isomorphism preserving edges test");

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //                0-----0           (output index 0 -> input index 0)
  //                         0-----0
  //                1--2              (output index 1 -> input index 2)
  //                         1--2

  edges = {{3, 5, 0, 0}, {6, 8, 0, 0}, {3, 4, 1, 2}, {6, 7, 1, 2}};

  expected_matches = {
      {{0, 1, 2}, 1}, // 0
      {{3, 6}, 1},    // 1
      {{4, 7}, 1},    // 2
      {{5, 8}, 1},    // 3
      {{0, 1}, 2},    // 0 0
      {{3, 6}, 2},    // 1 2
      {{4, 7}, 2},    // 2 3
      {{3, 6}, 3}     // 1 2 3
  };
  test(types, edges, expected_matches);

  // ----------------------------------------------------------
  popart::logging::info("differing input indices test");
  // in this test, edges break isomorphism because of input indices
  // This should actually throw an error, perhaps in the future it will

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //                0-----0           (output index 0 -> input index 0)
  //                         0-----1  (output index 0 -> input index 1)

  edges = {{3, 5, 0, 0}, {6, 8, 0, 1}};

  // no isomorphisms with the 3s
  expected_matches = {
      {{0, 1, 2}, 1}, // 0
      {{3, 6}, 1},    // 1
      {{4, 7}, 1},    // 2
      {{0, 1}, 2},    // 0 0
      {{3, 6}, 2},    // 1 2
  };
  test(types, edges, expected_matches);

  // ----------------------------------------------------------
  popart::logging::info("differing internal output indices test");
  // in this test, edges break isomorphism because of
  // differening internal output indices

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //                0-----0           (output index 0 -> input index 0)
  //                         1-----0  (output index 0 -> input index 1)

  edges = {{3, 5, 0, 0}, {6, 8, 1, 0}};

  expected_matches = {
      {{0, 1, 2}, 1}, // 0
      {{4, 7}, 1},    // 2
      {{5, 8}, 1},    // 3
      {{0, 1}, 2},    // 0 0
      {{4, 7}, 2}     // 2 3
                      // no 1 2 3
  };
  test(types, edges, expected_matches);

  // ----------------------------------------------------------
  popart::logging::info("internal vs external input test");
  // in this test, the [123]s are not isomporphic as one has
  // an internal input to 3, the other has an external input

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //                0-----0           (output index 0 -> input index 0)
  //             0-----------------0  (output index 0 -> input index 0)

  edges = {{3, 5, 0, 0}, {2, 8, 0, 0}};

  expected_matches = {
      {{0, 1}, 1}, // 0
      {{4, 7}, 1}, // 2
      {{5, 8}, 1}, // 3
      {{4, 7}, 2}  // 2 3
                   // no 1 2 3
  };
  test(types, edges, expected_matches);

  // -------------------------------------------------------------------
  popart::logging::info("isomorphically different external input test");
  // in this test, the [123]s are not isomporphic as one has
  // an internal input to 3, the other has an external input

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //          0-----0
  //          0--------0
  //          0-----------0
  //       0-----------------0
  //       0--------------------0
  //             0-----------------0

  edges = {{1, 3, 0, 0},
           {1, 4, 0, 0},
           {1, 5, 0, 0},
           {0, 6, 0, 0},
           {0, 7, 0, 0},
           {2, 8, 0, 0}};

  expected_matches = {
      {{0, 1, 2}, 1}, // 0
      {{3, 6}, 1},    // 1
      {{4, 7}, 1},    // 2
      {{5, 8}, 1},    // 3
      {{0, 1}, 2},    // 0 0
      {{3, 6}, 2}     // 1 2
                      // no 2 3
                      // no 1 2 3
  };
  test(types, edges, expected_matches);

  // -------------------------------------------------------------------
  popart::logging::info("non-isomorphically identical external input test");
  // in this test, the [123]s are not isomorphic as one has
  // an internal input to 3, the other has an external input

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //          0-----0
  //          0--------1
  //          0-----------2
  //       1-----------------0
  //       1--------------------1
  //       1-----------------------2

  edges = {{1, 3, 0, 0},
           {1, 4, 0, 1},
           {1, 5, 0, 2},
           {0, 6, 1, 0},
           {0, 7, 1, 1},
           {0, 8, 1, 2}};

  expected_matches = {
      {{3, 6}, 1}, // 1
      {{4, 7}, 1}, // 2
      {{5, 8}, 1}, // 3
      {{3, 6}, 2}, // 1 2
      {{4, 7}, 2}, // 2 3
      {{3, 6}, 3}  // 1 2 3
  };
  test(types, edges, expected_matches);

  // --------------------------------------------------------------------------
  popart::logging::info("isomorphically different external output index test");
  // in this test, the [123]s are not isomporphic as one has
  // an internal input to 3, the other has an external input

  //       0  1  2  3  4  5  6  7  8
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3};
  //          0-----0
  //          0--------1
  //          0-----------2
  //       1-----------------0
  //       2--------------------1
  //       1-----------------------2

  edges = {{1, 3, 0, 0},
           {1, 4, 0, 1},
           {1, 5, 0, 2},
           {0, 6, 1, 0},
           {0, 7, 2, 1},
           {0, 8, 1, 2}};

  expected_matches = {
      {{3, 6}, 1}, // 1
      {{4, 7}, 1}, // 2
      {{5, 8}, 1}, // 3
                   // no 1 2
                   // no 2 3
                   // no 1 2 3
  };
  test(types, edges, expected_matches);

  // --------------------------------------------------------------------------
  popart::logging::info("four isomorphically equivalent");
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  //                0-----1
  //                   0--2
  //                         0-----1
  //                            0--2
  //                                  0-----1
  //                                     0--2
  //                                           0-----1
  //                                              0--2
  edges = {};
  for (int i = 0; i < 4; ++i) {
    edges.push_back({3 + 3 * i, 5 + 3 * i, 0, 1});
    edges.push_back({4 + 3 * i, 5 + 3 * i, 0, 2});
  }
  expected_matches = {
      {{0, 1, 2}, 1},      // 0
      {{3, 6, 9, 12}, 1},  // 1
      {{4, 7, 10, 13}, 1}, // 2
      {{5, 8, 11, 14}, 1}, // 3
      {{0, 1}, 2},         // 0 0
      {{3, 6, 9, 12}, 2},  // 1 2
      {{4, 7, 10, 13}, 2}, // 2 3
      {{5, 8, 11}, 2},     // 3 1
      {{3, 6, 9, 12}, 3},  // 1 2 3
      {{4, 7, 10}, 3},     // 2 3 1
      {{5, 8, 11}, 3},     // 3 1 2
      {{3, 6, 9}, 4},      // 1 2 3 1
      {{4, 7, 10}, 4},     // 2 3 1 2
      {{5, 8, 11}, 4},     // 3 1 2 3
      {{3, 6, 9}, 5},      // 1 2 3 1 2
      {{4, 7, 10}, 5},     // 2 3 1 2 3
      {{5, 8}, 5},         // 3 1 2 3 1
      {{3, 6, 9}, 6},      // 1 2 3 1 2 3
      {{4, 7}, 6},         // 2 3 1 2 3 1
      {{5, 8}, 6},         // 3 1 2 3 1 2
      {{3, 6}, 7},         // 1 2 3 1 2 3 1
      {{4, 7}, 7},         // 2 3 1 2 3 1 2
      {{5, 8}, 7},         // 3 1 2 3 1 2 3
      {{3, 6}, 8},         // 1 2 3 1 2 3 1 2
      {{4, 7}, 8},         // 2 3 1 2 3 1 2 3
      {{3, 6}, 9}          // 1 2 3 1 2 3 1 2 3
  };

  test(types, edges, expected_matches);

  // --------------------------------------------------------------------------
  popart::logging::info("two isomorphically equivalent");
  //       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
  types = {0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3};
  //                0-----1
  //                1-----0
  //                         0-----1
  //                         1-----0
  //                                  0-----0
  //                                  1-----1
  //                                           0-----0
  //                                           1-----1
  edges = {};
  for (int i = 0; i < 4; ++i) {
    int switch_up = i < 2 ? 1 : 0;
    edges.push_back({3 + 3 * i, 5 + 3 * i, 1 - switch_up, 1});
    edges.push_back({3 + 3 * i, 5 + 3 * i, switch_up, 0});
  }

  expected_matches = {
      {{0, 1, 2}, 1},      // 0
      {{3, 6, 9, 12}, 1},  // 1
      {{4, 7, 10, 13}, 1}, // 2
      {{5, 8, 11, 14}, 1}, // 3 with different source tensors
      {{0, 1}, 2},         // 0 0
      {{3, 6, 9, 12}, 2},  // 1 2
      {{4, 7, 10, 13}, 2}, // 2 3 with different source
      {{5, 8, 11}, 2},
      {{3, 6}, 3},     // 1 2 3
      {{9, 12}, 3},    // 1 2 3 with different source
      {{4, 7, 10}, 3}, // 2 3 1
      {{5, 8, 11}, 3}, // 3 1 2
      {{3, 6}, 4},     // 1 2 3 1
      {{4, 7, 10}, 4}, // 2 3 1 2
      {{8, 11}, 4},    // 3 1 2 3
      {{3, 6}, 5},     // 1 2 3 1 2
      {{7, 10}, 5}     // 2 3 1 2 3
  };

  test(types, edges, expected_matches);

  // --------------------------------------------------------------------------
  popart::logging::info("iso different, because different output consumed");
  //       0  1  2  3  4    5    6  7  8  9
  types = {0, 1, 2, 3, 100, 101, 1, 2, 3, 102};
  //          0--------1         1--------1
  edges            = {{1, 4, 0, 1}, {6, 9, 1, 1}};
  expected_matches = {{{2, 7}, 1}, {{3, 8}, 1}, {{2, 7}, 2}};
  test(types, edges, expected_matches);
}
