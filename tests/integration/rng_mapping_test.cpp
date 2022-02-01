// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RngMappingTest

#include <boost/test/unit_test.hpp>

#include <poplar/Target.hpp>

#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>

#include "../../willow/src/popx/rng/rngstatelowering.hpp"

// This is the function that was previously in rngstatelowering, for setting the
// layout of the RNG state tensors.
void testLayoutRNGStateTensor(snap::Graph &graph, snap::Tensor &tensor) {

  auto numTiles = graph.getTarget().getNumTiles();
  if (tensor.rank() >= 1 && tensor.dim(0) == numTiles) {

    for (auto tile = 0U; tile != numTiles; ++tile) {
      auto slice = tensor.slice({tile, tile + 1}, 0);
      graph.getPoplarGraph().setTileMapping(slice.getPoplarTensor(), tile);
    }

  } else {
    throw popart::internal_error(
        "[RngStateLowering] Expected tensor with first "
        "dimension of {} (got tensor shape {})",
        numTiles,
        tensor.shape());
  }
}

// Derive popart::popx::RngStateLowering to provide public access to
// createRNGStateTensor and layoutRNGStateTensor.
class RngStateLoweringLayoutTester : public popart::popx::RngStateLowering {
public:
  static snap::Tensor createStateTensor(snap::Graph &graph) {
    return createRNGStateTensor(graph, "");
  }

  static void layoutStateTensor(snap::Graph &graph, snap::Tensor &tensor) {
    layoutRNGStateTensor(graph, tensor);
  }
};

// This test checks that popart::popx::layoutRNGStateTensor produces the
// correct layout. As layoutRNGStateTensor uses poplibs internally, so this
// test will give us a warning if the poplibs implementation changes at all.
BOOST_AUTO_TEST_CASE(TestRngMapping) {
  auto target = poplar::Target::createIPUTarget(1, "ipu2");
  snap::Graph graph(target);

  auto tensorWithExpectedLayout =
      RngStateLoweringLayoutTester::createStateTensor(graph);
  testLayoutRNGStateTensor(graph, tensorWithExpectedLayout);
  auto expectedLayout = graph.getPoplarGraph().getTileMapping(
      tensorWithExpectedLayout.getPoplarTensor());

  auto tensorWithActualLayout =
      RngStateLoweringLayoutTester::createStateTensor(graph);
  RngStateLoweringLayoutTester::layoutStateTensor(graph,
                                                  tensorWithActualLayout);
  auto actualLayout = graph.getPoplarGraph().getTileMapping(
      tensorWithActualLayout.getPoplarTensor());

  BOOST_REQUIRE_EQUAL(actualLayout, expectedLayout);
}
