// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE unittest_backwardsgraphcreatorhelper

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>

#define private public
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>
#undef private

using namespace popart;

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdIdIsGrad) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  FwdGraphToBwdGraphInfo gradInfo;
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph, gradInfo};

  BOOST_REQUIRE(!helper.bwdIdIsGrad(bwdGraph.addScope("t0")));
  BOOST_REQUIRE(helper.bwdIdIsGrad(bwdGraph.addScope(getGradId("t0"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdIdIsNonGrad) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  FwdGraphToBwdGraphInfo gradInfo;
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph, gradInfo};

  BOOST_REQUIRE(helper.bwdIdIsNonGrad(bwdGraph.addScope("t0")));
  BOOST_REQUIRE(!helper.bwdIdIsNonGrad(bwdGraph.addScope(getGradId("t0"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_fwdIdToBwdGradId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  FwdGraphToBwdGraphInfo gradInfo;
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph, gradInfo};

  BOOST_REQUIRE(bwdGraph.addScope(getGradId("t2")) ==
                helper.fwdIdToBwdGradId(fwdGraph.addScope("t2")));
}
BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdGradIdToFwdId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  FwdGraphToBwdGraphInfo gradInfo;
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph, gradInfo};

  BOOST_REQUIRE(fwdGraph.addScope("t2") ==
                helper.bwdGradIdToFwdId(bwdGraph.addScope(getGradId("t2"))));
}

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_bwdNonGradIdToFwdId) {

  Ir ir;
  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  FwdGraphToBwdGraphInfo gradInfo;
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph, gradInfo};

  BOOST_REQUIRE(fwdGraph.addScope("t2") ==
                helper.bwdNonGradIdToFwdId(bwdGraph.addScope("t2")));
}