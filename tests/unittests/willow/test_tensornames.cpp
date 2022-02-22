// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UnittestWillowTensorNames
#include <boost/test/unit_test.hpp>

#include <popart/tensornames.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Test_fwdIdToBwdGradId) {
  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));

  const auto bwdGradId = addScope(bwdGraph, getGradId("t2"));
  const auto fwdId     = addScope(fwdGraph, "t2");

  BOOST_REQUIRE(bwdGradId == fwdIdToBwdGradId(fwdGraph, bwdGraph, fwdId));
}

BOOST_AUTO_TEST_CASE(Test_bwdGradIdToFwdId) {
  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));

  const auto fwdId     = addScope(fwdGraph, "t2");
  const auto bwdGradId = addScope(bwdGraph, getGradId("t2"));

  BOOST_REQUIRE(fwdId == bwdGradIdToFwdId(fwdGraph, bwdGraph, bwdGradId));
}

BOOST_AUTO_TEST_CASE(Test_bwdNonGradIdToFwdId) {
  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));

  const auto fwdId        = addScope(fwdGraph, "t2");
  const auto bwdNonGradId = addScope(bwdGraph, "t2");

  BOOST_REQUIRE(fwdId == bwdNonGradIdToFwdId(fwdGraph, bwdGraph, bwdNonGradId));
}
