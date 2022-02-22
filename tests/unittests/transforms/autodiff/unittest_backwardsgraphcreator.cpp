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
  auto &fwdGraph = ir.createGraph(GraphId("A"));

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

/**
 * Calling BackwardsGraphCreator::createBackwardsGraph with a bwd graph id that
 * already exists should throw.
 */
BOOST_AUTO_TEST_CASE(
    backwardsgraphcreator_createBackwardsGraph_with_bwd_graph_id_that_already_exists_throws) {
  const auto ti = TensorInfo{DataType::FLOAT, Shape{2}};

  Ir ir;

  Graph &mg     = ir.getMainGraph();
  TensorId mg_t = "t";
  std::vector<float> mg_t_data(ti.nelms());
  mg.getTensors().addVarInit(mg_t, ti, mg_t_data.data());

  AutodiffIrAdapter adapter{ir};
  BackwardsGraphCreator creator{adapter};

  const GraphId bwdGraphId = creator.genNewBwdGraphId(mg.id);

  auto callCreateBackwardsGraph = [&]() {
    return creator.createBackwardsGraph(
        mg,
        bwdGraphId, // Same bwdGraphId every time
        BackwardsGraphCreator::TensorIds({mg_t}),
        nonstd::nullopt,
        {});
  };

  BOOST_REQUIRE_NO_THROW(callCreateBackwardsGraph());
  BOOST_REQUIRE_THROW(callCreateBackwardsGraph(), popart::internal_error);
}
