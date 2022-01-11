// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_backwardsgraphcreator

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/backwardsgraphcreator.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(backwardsgraphcreator_genNewBwdGraphId) {

  // Check genNewBwdGraphId helper function generates unique IDs.

  Ir ir;
  auto &fwdGraph  = ir.createGraph(GraphId("A"));

  AutodiffIrAdapter adapter{ir};
  BackwardsGraphCreator creator{adapter};

  auto id0 = creator.genNewBwdGraphId(fwdGraph.id);
  ir.createGraph(id0);
  auto id1 = creator.genNewBwdGraphId(fwdGraph.id);
  ir.createGraph(id1);
  auto id2 = creator.genNewBwdGraphId(fwdGraph.id);

  BOOST_REQUIRE(id0 != id1);
  BOOST_REQUIRE(id1 != id2);
  BOOST_REQUIRE(id0 != id2);
}
