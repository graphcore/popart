// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE MaxCliqueTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <memory>
#include <vector>
#include <popart/maxclique.hpp>

using namespace popart;
using namespace popart::graphclique;

BOOST_AUTO_TEST_CASE(MaxCliqueTest_0) {

  graphclique::AGraph ag(7);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      ag.addEdge(i, j);
    }
  }

  for (int i = 4; i < 7; ++i) {
    for (int j = 4; j < 7; ++j) {
      ag.addEdge(i, j);
    }
  }

  graphclique::MaxClique mq(ag);

  auto mcliques = mq.getMaximumCliques(1, ag.numVertices());
  BOOST_ASSERT(mcliques.size() == 2);
  for (int i = 0; i < 5; ++i) {
    BOOST_ASSERT(std::find(mcliques[0].begin(), mcliques[0].end(), i) !=
                 mcliques[0].end());
  }
  for (int i = 5; i < 7; ++i) {
    BOOST_ASSERT(std::find(mcliques[1].begin(), mcliques[1].end(), i) !=
                 mcliques[1].end());
  }
}
