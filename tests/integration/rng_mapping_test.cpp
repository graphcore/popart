// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RngMappingTest

#include <boost/test/unit_test.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>

#include <testdevice.hpp>

#include "../../willow/src/popx/rng/rngstatelowering.hpp"
#include "popart/error.hpp"
#include "popart/util.hpp" // IWYU pragma: keep

// This is the function that was previously in rngstatelowering, for setting the
// layout of the RNG state tensors.
void testLayoutRNGStateTensor(poplar::Graph &graph, poplar::Tensor &tensor) {

  auto numTiles = graph.getTarget().getNumTiles();
  if (tensor.rank() >= 1 && tensor.dim(0) == numTiles) {

    for (auto tile = 0U; tile != numTiles; ++tile) {
      auto slice = tensor.slice({tile, tile + 1}, 0);
      graph.setTileMapping(slice, tile);
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
  static poplar::Tensor createStateTensor(poplar::Graph &graph) {
    return createRNGStateTensor(graph, "");
  }
};

// This test checks that popart::popx::layoutRNGStateTensor produces the
// correct layout. As layoutRNGStateTensor uses poplibs internally, so this
// test will give us a warning if the poplibs implementation changes at all.
BOOST_AUTO_TEST_CASE(TestRngMapping) {
  auto target = poplar::Target::createIPUTarget(1, "ipu2");
  poplar::Graph graph(target);

  auto t = RngStateLoweringLayoutTester::createStateTensor(graph);

  // This is the actual layout that is used in popart.
  auto actualLayout = graph.getTileMapping(t);

  testLayoutRNGStateTensor(graph, t);

  // This is the layout we want to have.
  auto expectedLayout = graph.getTileMapping(t);

  BOOST_REQUIRE_EQUAL(actualLayout, expectedLayout);
}

BOOST_AUTO_TEST_CASE(RngStateTensorSizeAndShapeConsistencyTest) {
  static constexpr unsigned numIpus   = 4;
  static constexpr unsigned repFactor = 5;

  const auto target =
      poplar::Target::createIPUTarget(numIpus * repFactor, "ipu2");
  const poplar::Graph graph(target, poplar::replication_factor(repFactor));

  BOOST_CHECK_EQUAL(numIpus * repFactor, target.getNumIPUs());
  BOOST_CHECK_EQUAL(numIpus, graph.getTarget().getNumIPUs());

  const auto deviceInfo =
      popart::createTestDevice(popart::TestDeviceType::IpuModel21,
                               target.getNumIPUs(),
                               target.getTilesPerIPU());

  const auto shapeGraph =
      popart::popx::RngStateLowering::getCombinedRngStateTensorShape(graph);
  const auto shapeDeviceInfo =
      popart::popx::RngStateLowering::getCombinedRngStateTensorShape(
          *deviceInfo, repFactor);

  const auto sizeGraph =
      popart::popx::RngStateLowering::getCombinedRngStateTensorSize(graph);
  const auto sizeDeviceInfo =
      popart::popx::RngStateLowering::getCombinedRngStateTensorSize(*deviceInfo,
                                                                    repFactor);

  BOOST_CHECK_EQUAL(shapeGraph, shapeDeviceInfo);
  BOOST_CHECK_EQUAL(sizeGraph, sizeDeviceInfo);
}
