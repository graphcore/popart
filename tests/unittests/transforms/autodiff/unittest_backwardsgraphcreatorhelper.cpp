// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE unittest_backwardsgraphcreatorhelper

#include <boost/test/unit_test.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
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

BOOST_AUTO_TEST_CASE(backwardsgraphcreatorhelper_growgradsum_inherit_vgid) {
  // Test to ensure that growing the GradSumOp for a backwards graph
  // includes inheriting virtual graph placement.
  Ir ir;
  ir.getSessionOptions().virtualGraphMode = VirtualGraphMode::Manual;

  Graph &fwdGraph = ir.createGraph(GraphId("fwd"));
  Graph &bwdGraph = ir.createGraph(GraphId("bwd"));
  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};

  std::vector<int32_t> data{1};
  fwdGraph.addConstInit(
      addScope(fwdGraph, "t1"), {DataType::INT32, {}}, data.data(), "");
  auto target = fwdGraph.getTensor(addScope(fwdGraph, "t1"));
  bwdGraph.addConstInit(
      addScope(fwdGraph, "t2"), {DataType::INT32, {}}, data.data(), "");

  auto settings = Op::Settings(bwdGraph, "id");
  auto op =
      bwdGraph.createConnectedOp<IdentityOp>({{0, addScope(fwdGraph, "t2")}},
                                             {{0, addScope(fwdGraph, "t3")}},
                                             Onnx::Operators::Identity_1,
                                             settings);
  op->setVirtualGraphId(1);
  auto partial = bwdGraph.getTensor(addScope(fwdGraph, "t3"));

  auto gradSum = helper.growGradSumOp(target, {partial});
  BOOST_REQUIRE(gradSum->hasVirtualGraphId());
  BOOST_REQUIRE(gradSum->getVirtualGraphId() == 1);
}