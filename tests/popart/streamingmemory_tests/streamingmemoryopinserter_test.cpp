
// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE StreamingMemoryOpInserterTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/scheduler.hpp>
#include <popart/topocons.hpp>

#include <transforms/streamingmemoryopinserter.hpp>

using namespace popart;
using namespace popart::io;

namespace {

// Need to derive from StreamingMemoryOpInserter to get access to certain
// functions.
class StreamingMemoryOpInserterTestWrapper : public StreamingMemoryOpInserter {
public:
  StreamingMemoryOpInserterTestWrapper(Graph &graph,
                                       int64_t replFactor,
                                       int numStages,
                                       int numPhases)
      : StreamingMemoryOpInserter{graph, replFactor, numStages, numPhases} {}

  void getTensorStreamingConfig(Tensor *tensor) {
    return StreamingMemoryOpInserter::getTensorStreamingConfig(tensor);
  }
};

const int64_t replFactor = 2;
const int64_t numStages  = 2;
const int64_t numPhases  = 8;
} // namespace

BOOST_AUTO_TEST_CASE(StreamingMemoryOpInserter_determineTensorLocation) {

  Ir ir;
  Graph graph = {ir, GraphId::root()};

  //   weight0    opt0
  //     |         |
  //     +--AddOp0-+
  //          |
  //         act0

  TensorId act0{"act0"};
  TensorId weight0{"weight0"};
  TensorId opt0{std::string(reservedAccumPrefix()) + "opt0"};
  TensorInfo tensorInfo0{"FLOAT", std::vector<int64_t>{10}};
  auto data = std::vector<float>(10, 0.0f);

  graph.getTensors().addVarInit(weight0, tensorInfo0, data.data());
  graph.getTensors().addVarInit(opt0, tensorInfo0, data.data());

  auto addOp0 =
      std::make_unique<AddOp>(Onnx::Operators::Add_7, Op::Settings{graph, ""});
  addOp0->id = 0;
  addOp0->connectInTensor(AddOp::getArg0InIndex(), weight0);
  addOp0->connectInTensor(AddOp::getArg1InIndex(), opt0);
  addOp0->createAndConnectOutTensor(AddOp::getOutIndex(), act0);
  addOp0->setup();
  graph.moveIntoGraph(std::move(addOp0));

  StreamingMemoryOpInserterTestWrapper inserter(
      graph, replFactor, numStages, numPhases);

  // With default settings everything is on chip.
  {
    SessionOptions options;
    options.enableReplicatedGraphs = true;
    options.replicatedGraphCount   = 2;
    ir.setUserOptions(options);

    auto act0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OnChip == act0Loc.storage);
    BOOST_CHECK(TileSet::Compute == act0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == act0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                act0Loc.replicatedTensorSharding);

    auto weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(TensorStorage::OnChip == weight0Loc.storage);
    BOOST_CHECK(TileSet::Compute == weight0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == weight0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                weight0Loc.replicatedTensorSharding);

    auto opt0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(opt0));
    BOOST_CHECK(TensorStorage::OnChip == act0Loc.storage);
    BOOST_CHECK(TileSet::Compute == act0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == act0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                act0Loc.replicatedTensorSharding);
  }

  // Check that settings to activationTensorLocationSettings are acted upon.
  {
    SessionOptions options;
    options.enableReplicatedGraphs = true;
    options.replicatedGraphCount   = 2;
    options.activationTensorLocationSettings.location.storage =
        TensorStorage::OffChip;
    options.activationTensorLocationSettings.location.loadTileSet = TileSet::IO;
    options.activationTensorLocationSettings.location.replicatedTensorSharding =
        ReplicatedTensorSharding::On;
    ir.setUserOptions(options);

    auto act0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OffChip == act0Loc.storage);
    BOOST_CHECK(TileSet::IO == act0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == act0Loc.storageTileSet);
    // Activations are currently never sharded.
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                act0Loc.replicatedTensorSharding);

    auto weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(TensorStorage::OnChip == weight0Loc.storage);
    BOOST_CHECK(TileSet::Compute == weight0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == weight0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                weight0Loc.replicatedTensorSharding);

    auto opt0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(opt0));
    BOOST_CHECK(TensorStorage::OnChip == opt0Loc.storage);
    BOOST_CHECK(TileSet::Compute == opt0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == opt0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                opt0Loc.replicatedTensorSharding);

    // Check that with a higher threshold the activation is not off-chipped.
    options.activationTensorLocationSettings.minElementsForOffChip = 16;
    ir.setUserOptions(options);
    act0Loc = inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OnChip == act0Loc.storage);
  }

  // Check that settings to weightTensorLocationSettings are acted upon.
  {
    SessionOptions options;
    options.enableReplicatedGraphs = true;
    options.replicatedGraphCount   = 2;
    options.weightTensorLocationSettings.location.storage =
        TensorStorage::OffChip;
    options.weightTensorLocationSettings.location.loadTileSet    = TileSet::IO;
    options.weightTensorLocationSettings.location.storageTileSet = TileSet::IO;
    options.weightTensorLocationSettings.location.replicatedTensorSharding =
        ReplicatedTensorSharding::On;
    options.weightTensorLocationSettings
        .minElementsForReplicatedTensorSharding = 6;
    ir.setUserOptions(options);

    auto act0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OnChip == act0Loc.storage);
    BOOST_CHECK(TileSet::Compute == act0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == act0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                act0Loc.replicatedTensorSharding);

    auto weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(TensorStorage::OffChip == weight0Loc.storage);
    BOOST_CHECK(TileSet::IO == weight0Loc.loadTileSet);
    BOOST_CHECK(TileSet::IO == weight0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::On ==
                weight0Loc.replicatedTensorSharding);

    auto opt0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(opt0));
    BOOST_CHECK(TensorStorage::OnChip == opt0Loc.storage);
    BOOST_CHECK(TileSet::Compute == opt0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == opt0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                opt0Loc.replicatedTensorSharding);

    // Check that with a higher threshold the weight tensor is not sharded.
    options.weightTensorLocationSettings
        .minElementsForReplicatedTensorSharding = 16;
    ir.setUserOptions(options);
    weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                weight0Loc.replicatedTensorSharding);

    // Check that with a higher threshold the weight is not off-chipped.
    options.weightTensorLocationSettings.minElementsForOffChip = 16;
    ir.setUserOptions(options);
    weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(TensorStorage::OnChip == weight0Loc.storage);
  }

  // Check that settings to optimizerStateTensorLocationSettings are acted upon.
  {
    SessionOptions options;
    options.enableReplicatedGraphs = true;
    options.replicatedGraphCount   = 2;
    options.optimizerStateTensorLocationSettings.location.storage =
        TensorStorage::OffChip;
    options.optimizerStateTensorLocationSettings.location.loadTileSet =
        TileSet::Compute;
    options.optimizerStateTensorLocationSettings.location.storageTileSet =
        TileSet::IO;
    options.optimizerStateTensorLocationSettings.location
        .replicatedTensorSharding = ReplicatedTensorSharding::Off;
    options.optimizerStateTensorLocationSettings
        .minElementsForReplicatedTensorSharding = 6;
    ir.setUserOptions(options);

    auto act0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OnChip == act0Loc.storage);
    BOOST_CHECK(TileSet::Compute == act0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == act0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                act0Loc.replicatedTensorSharding);

    auto weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(TensorStorage::OnChip == weight0Loc.storage);
    BOOST_CHECK(TileSet::Compute == weight0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == weight0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                weight0Loc.replicatedTensorSharding);

    auto opt0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(opt0));
    BOOST_CHECK(TensorStorage::OnChip == opt0Loc.storage);
    BOOST_CHECK(TileSet::Compute == opt0Loc.loadTileSet);
    BOOST_CHECK(TileSet::Compute == opt0Loc.storageTileSet);
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                opt0Loc.replicatedTensorSharding);

    // Check that with a higher threshold the opt state tensor is not sharded.
    options.optimizerStateTensorLocationSettings
        .minElementsForReplicatedTensorSharding = 16;
    ir.setUserOptions(options);
    weight0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(weight0));
    BOOST_CHECK(ReplicatedTensorSharding::Off ==
                weight0Loc.replicatedTensorSharding);

    // Check that with a higher threshold the opt state is not off-chipped.
    options.optimizerStateTensorLocationSettings.minElementsForOffChip = 16;
    ir.setUserOptions(options);
    opt0Loc = inserter.determineTensorLocation(graph.getTensors().get(opt0));
    BOOST_CHECK(TensorStorage::OnChip == opt0Loc.storage);
  }

  // Check we can make exceptions for single tensors.
  {
    SessionOptions options;
    options.tensorLocationSettingsOverride[act0] = TensorStorage::OffChip;
    // options.tensorLocationSettingsOverride[act0].loadTileSet = TileSet::IO;
    options.enableReplicatedGraphs = true;
    options.replicatedGraphCount   = 2;
    ir.setUserOptions(options);

    auto act0Loc =
        inserter.determineTensorLocation(graph.getTensors().get(act0));
    BOOST_CHECK(TensorStorage::OffChip == act0Loc.storage);
  }
}