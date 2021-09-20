// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE unittest_backwardsgraphcreatorhelper

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/util.hpp>

#define private public
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>
#undef private

using namespace popart;

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdIdIsGrad) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  BOOST_REQUIRE(!helper.bwdIdIsGrad(addScope(bwdGraph, "t0")));
  BOOST_REQUIRE(helper.bwdIdIsGrad(addScope(bwdGraph, getGradId("t0"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdIdIsNonGrad) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  BOOST_REQUIRE(helper.bwdIdIsNonGrad(addScope(bwdGraph, "t0")));
  BOOST_REQUIRE(!helper.bwdIdIsNonGrad(addScope(bwdGraph, getGradId("t0"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_fwdIdToBwdGradId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  BOOST_REQUIRE(addScope(bwdGraph, getGradId("t2")) ==
                helper.fwdIdToBwdGradId(addScope(fwdGraph, "t2")));
}
BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdGradIdToFwdId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  BOOST_REQUIRE(addScope(fwdGraph, "t2") ==
                helper.bwdGradIdToFwdId(addScope(bwdGraph, getGradId("t2"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdNonGradIdToFwdId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  BOOST_REQUIRE(addScope(fwdGraph, "t2") ==
                helper.bwdNonGradIdToFwdId(addScope(bwdGraph, "t2")));
}