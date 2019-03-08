#define BOOST_TEST_MODULE Speed0SubgraphTest

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "blip.hpp"

using namespace fwtools::subgraph;
using namespace blip;

BOOST_AUTO_TEST_CASE(Speed0_Subgraph) {

  int n_nodes = 10000;
  int seed    = 1011;
  std::default_random_engine eng(seed);
  std::uniform_int_distribution<int> idis(0, 5);

  std::vector<const Blip *> sched;
  std::vector<std::unique_ptr<Blip>> blips;

  for (int i = 0; i < n_nodes; ++i) {
    blips.emplace_back(std::unique_ptr<Blip>(new Blip(idis(eng), 10.0f, {})));
    sched.push_back(blips[i].get());
  }

  RinseMatcher<const Blip> rinseMatcher(sched);
  auto matches0 = rinseMatcher.getRepeatedSequences();
  auto matches1 = rinseMatcher.separateByIsomorphism(matches0);

  // O(1 second) for 100,000 nodes.
  // TODO: include some time header and verify that this is fast
  // T7258
  BOOST_CHECK(1 == 1);
}
