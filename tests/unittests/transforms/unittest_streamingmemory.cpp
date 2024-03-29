// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE streamingmemory_unittest

#include <boost/test/unit_test.hpp>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <testutil/test_graphs/graph_test_models.hpp>
#include <vector>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/init.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/scaledadd.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1acclupdate.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/sgd2acclupdate.hpp>
#include <popart/op/sgd2varupdate.hpp>
#include <popart/transforms/prune.hpp>
#include <popart/transforms/streamingmemory.hpp>

#include "popart/commgroup.hpp"
#include "popart/devicemanager.hpp"
#include "popart/error.hpp"
#include "popart/graph.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/vendored/optional.hpp"
#include "testutil/irquery/irquery.hpp"

using namespace popart;
using namespace popart::irquery;

// Test if weights and optimizer states are sharded correctly with different
// optimizers
BOOST_AUTO_TEST_CASE(BasicReplicatedTensorShardingTest) {
  SessionOptions options;

  options.enableReplicatedGraphs            = true;
  options.enableDistributedReplicatedGraphs = false;
  options.replicatedGraphCount              = 4;
  options.globalReplicaOffset               = 0;
  options.globalReplicationFactor           = 1;
  options.weightTensorLocationSettings.minElementsForReplicatedTensorSharding =
      4;
  options.weightTensorLocationSettings.location.replicatedTensorSharding =
      ReplicatedTensorSharding::On;
  options.optimizerStateTensorLocationSettings
      .minElementsForReplicatedTensorSharding = 4;
  options.optimizerStateTensorLocationSettings.location
      .replicatedTensorSharding = ReplicatedTensorSharding::On;

  std::set<TestOptimizer> testOptimizers{TestOptimizer::SGD0,
                                         TestOptimizer::SGD1,
                                         TestOptimizer::SGD2,
                                         TestOptimizer::Adam,
                                         TestOptimizer::Lamb};

  for (auto testOptimizer : testOptimizers) {
    OptimizerTestModel model(testOptimizer, 1, options);

    auto &ir = model.getIr();

    auto &dm    = DeviceManager::createDeviceManager();
    auto device = dm.createOfflineIPUDevice({{"numIPUs", "4"}});
    BOOST_REQUIRE(device);
    ir.setDeviceInfo(*device);

    ir.applyTransform(StreamingMemory::id(1), ir.getMainGraph());
    ir.applyTransform(StreamingMemory::id(2), ir.getMainGraph());
    ir.updateVertices();
    ir.applyTransform(Prune::id(), ir.getMainGraph());
    ir.updateVertices();
    ir.setIsPrepared();

    ir.dotCheckpoint(ir, "Final");

    IrTestWrapper tw_ir{ir};
    auto tw_mainGraph =
        tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
    auto &graph = tw_mainGraph->unwrap().get();

    const auto aGroupOfTypeAll =
        ReplicaGrouping(options.getGlobalReplicationFactor());

    switch (testOptimizer) {
    case TestOptimizer::SGD0: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [](const Op *op) { return op->isConvertibleTo<SGD0VarUpdateOp>(); }};
      graphutils::Edges edges{{0, 1}, {1, 2}, {1, 4}, {3, 4}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::SGD1: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD1AcclUpdateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD1VarUpdateOp>(); }};
      graphutils::Edges edges{
          {0, 2}, {1, 3}, {2, 4}, {2, 8}, {3, 6}, {5, 6}, {6, 7}, {6, 8}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::SGD2: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          },
          [](const Op *op) {
            return op->isConvertibleTo<SGD2PartialAcclUpdateOp>();
          },
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD2VarUpdateOp>(); }};
      graphutils::Edges edges{{0, 2},
                              {1, 3},
                              {2, 6},
                              {2, 4},
                              {2, 8},
                              {3, 6, 0, 0},
                              {5, 7},
                              {6, 7},
                              {7, 8}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::Adam: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 0
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 1
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 2
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 3
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 4
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 5
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          }, // 6
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          }, // 7
          [](const Op *op) {
            return op->isConvertibleTo<AdamUpdaterOp>();
          },                                                                // 8
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); }, // 9
          [](const Op *op) {
            return op->isConvertibleTo<AccumulateOp>();
          }, // 10
          [](const Op *op) {
            return op->isConvertibleTo<AdamVarUpdateOp>();
          },                                                              // 11
          [](const Op *op) { return op->isConvertibleTo<ScaleOp>(); },    // 12
          [](const Op *op) { return op->isConvertibleTo<ScaledAddOp>(); } // 13
      };
      graphutils::Edges edges{
          {0, 3},  // Init -> RemoteLoad
          {1, 4},  // Init -> RemoteLoad
          {2, 5},  // Init -> RemoteLoad
          {3, 11}, // RemoteLoad -> AdamVarUpdateOp
          {3, 6},  // RemoteLoad -> ReplicatedAllGatherOp
          {4, 9},  // RemoteLoad -> AccumulateOp
          {5, 10}, // RemoteLoad -> AccumulateOp
          {9,
           8,
           0,
           AdamUpdaterOp::getAccl1InIndex()}, // Accumulate -> AdamUpdater
          {10,
           8,
           0,
           AdamUpdaterOp::getAccl2InIndex()}, // Accumulate -> AdamUpdater
          {8, 11},                            // AdamUpdater -> AdamVarUpdateOp
          {7, 12},  // ReplicatedReduceScatterOp -> Scale
          {12, 13}, // Scale -> ScaledAdd
          {13, 9},  // ScaledAdd -> Accumulate
          {13, 10}  // ScaledAdd -> Accumulate
      };

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::Lamb: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 0
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 1
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 2
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 3
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 4
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 5
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          }, // 6
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          }, // 7
          [](const Op *op) {
            return op->isConvertibleTo<AdamUpdaterOp>();
          },                                                                // 8
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); }, // 9
          [](const Op *op) {
            return op->isConvertibleTo<AccumulateOp>();
          }, // 10
          [](const Op *op) {
            return op->isConvertibleTo<AdamVarUpdateOp>();
          },                                                               // 11
          [](const Op *op) { return op->isConvertibleTo<ScaleOp>(); },     // 12
          [](const Op *op) { return op->isConvertibleTo<ScaledAddOp>(); }, // 13
          [](const Op *op) {
            return op->isConvertibleTo<LambSquareOp>();
          }, // 14
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceInplaceOp *>(op);
            return rop && rop->getReplicaGrouping() == aGroupOfTypeAll;
          } // 15
      };
      graphutils::Edges edges{
          {0, 3},  // Init -> RemoteLoad
          {1, 4},  // Init -> RemoteLoad
          {2, 5},  // Init -> RemoteLoad
          {3, 11}, // RemoteLoad -> AdamVarUpdateOp
          {3, 6},  // RemoteLoad -> ReplicatedAllGatherOp
          {4, 9},  // RemoteLoad -> AccumulateOp
          {5, 10}, // RemoteLoad -> AccumulateOp
          {9,
           8,
           0,
           AdamUpdaterOp::getAccl1InIndex()}, // Accumulate -> AdamUpdater
          {10,
           8,
           0,
           AdamUpdaterOp::getAccl2InIndex()}, // Accumulate -> AdamUpdater
          {8, 11},                            // AdamUpdater -> AdamVarUpdateOp
          {7, 12},  // ReplicatedReduceScatterOp -> Scale
          {12, 13}, // Scale -> ScaledAdd
          {13, 9},  // ScaledAdd -> Accumulate
          {13, 10}, // ScaledAdd -> Accumulate
          {8, 14},  // AdamUpdater -> LambSquare
          {14, 15}, // LambSquare -> ReplicatedAllReduceInplace
          {15, 11}, // ReplicatedAllReduceInplace -> AdamVarUpdate
      };

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    default: {
      throw error("Optimizer not supported by test");
      break;
    };
    }
  }
}

// Test if weights and optimizer states are sharded correctly with different
// optimizers split across two GCDs
BOOST_AUTO_TEST_CASE(DistributedReplicatedTensorShardingTest) {
  SessionOptions options;

  options.enableReplicatedGraphs            = true;
  options.enableDistributedReplicatedGraphs = true;
  options.replicatedGraphCount              = 4;
  options.globalReplicaOffset               = 0;
  options.globalReplicationFactor           = 8;

  const auto shardDomainSize = options.replicatedGraphCount;

  options.weightTensorLocationSettings.minElementsForReplicatedTensorSharding =
      4;
  options.weightTensorLocationSettings.location.replicatedTensorSharding =
      ReplicatedTensorSharding::On;
  options.weightTensorLocationSettings.location.shardingDomain =
      CommGroup(CommGroupType::Consecutive, shardDomainSize);
  options.optimizerStateTensorLocationSettings
      .minElementsForReplicatedTensorSharding = 4;
  options.optimizerStateTensorLocationSettings.location
      .replicatedTensorSharding = ReplicatedTensorSharding::On;
  options.optimizerStateTensorLocationSettings.location.shardingDomain =
      CommGroup(CommGroupType::Consecutive, shardDomainSize);

  std::set<TestOptimizer> testOptimizers{TestOptimizer::SGD0,
                                         TestOptimizer::SGD1,
                                         TestOptimizer::SGD2,
                                         TestOptimizer::Adam,
                                         TestOptimizer::Lamb};

  for (auto testOptimizer : testOptimizers) {
    OptimizerTestModel model(testOptimizer, 1, options);

    auto &ir = model.getIr();

    auto &dm    = DeviceManager::createDeviceManager();
    auto device = dm.createOfflineIPUDevice({{"numIPUs", "4"}});
    BOOST_REQUIRE(device);
    ir.setDeviceInfo(*device);

    ir.applyTransform(StreamingMemory::id(1), ir.getMainGraph());
    ir.applyTransform(StreamingMemory::id(2), ir.getMainGraph());
    ir.updateVertices();
    ir.applyTransform(Prune::id(), ir.getMainGraph());
    ir.updateVertices();
    ir.setIsPrepared();

    ir.dotCheckpoint(ir, "Final");

    IrTestWrapper tw_ir{ir};
    auto tw_mainGraph =
        tw_ir.hasGraph(ir.getMainGraph().id, Require::MustBeTrue);
    auto &graph = tw_mainGraph->unwrap().get();

    auto orthogonalGroupSize =
        options.getGlobalReplicationFactor() / shardDomainSize;
    auto orthogonalStride = shardDomainSize;

    const auto aConsecutiveGroup = ReplicaGrouping(
        options.getGlobalReplicationFactor(), 1, shardDomainSize);

    const auto anOrthogonalGroup =
        ReplicaGrouping(options.getGlobalReplicationFactor(),
                        orthogonalStride,
                        orthogonalGroupSize);

    switch (testOptimizer) {
    case TestOptimizer::SGD0: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceOp *>(op);
            return rop && rop->getReplicaGrouping() == anOrthogonalGroup;
          },
          [](const Op *op) { return op->isConvertibleTo<SGD0VarUpdateOp>(); }};
      graphutils::Edges edges{{0, 1}, {1, 2}, {1, 5}, {3, 4}, {4, 5}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::SGD1: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceOp *>(op);
            return rop && rop->getReplicaGrouping() == anOrthogonalGroup;
          },
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD1AcclUpdateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD1VarUpdateOp>(); }};
      graphutils::Edges edges{{0, 2},
                              {1, 3},
                              {2, 4},
                              {2, 9},
                              {3, 7},
                              {5, 6},
                              {6, 7},
                              {7, 8},
                              {7, 9}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::SGD2: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          },
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceOp *>(op);
            return rop && rop->getReplicaGrouping() == anOrthogonalGroup;
          },
          [](const Op *op) {
            return op->isConvertibleTo<SGD2PartialAcclUpdateOp>();
          },
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); },
          [](const Op *op) { return op->isConvertibleTo<SGD2VarUpdateOp>(); }};
      graphutils::Edges edges{{0, 2},
                              {1, 3},
                              {2, 7},
                              {2, 4},
                              {2, 9},
                              {3, 7, 0, 0},
                              {5, 6},
                              {6, 8},
                              {7, 8},
                              {8, 9}};

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::Adam: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 0
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 1
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 2
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 3
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 4
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 5
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          }, // 6
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          }, // 7
          [](const Op *op) {
            return op->isConvertibleTo<AdamUpdaterOp>();
          },                                                                // 8
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); }, // 9
          [](const Op *op) {
            return op->isConvertibleTo<AccumulateOp>();
          }, // 10
          [](const Op *op) {
            return op->isConvertibleTo<AdamVarUpdateOp>();
          },                                                               // 11
          [](const Op *op) { return op->isConvertibleTo<ScaleOp>(); },     // 12
          [](const Op *op) { return op->isConvertibleTo<ScaledAddOp>(); }, // 13
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceOp *>(op);
            return rop && rop->getReplicaGrouping() == anOrthogonalGroup;
          } // 14
      };
      graphutils::Edges edges{
          {0, 3},  // Init -> RemoteLoad
          {1, 4},  // Init -> RemoteLoad
          {2, 5},  // Init -> RemoteLoad
          {3, 11}, // RemoteLoad -> AdamVarUpdateOp
          {3, 6},  // RemoteLoad -> ReplicatedAllGatherOp
          {4, 9},  // RemoteLoad -> AccumulateOp
          {5, 10}, // RemoteLoad -> AccumulateOp
          {9,
           8,
           0,
           AdamUpdaterOp::getAccl1InIndex()}, // Accumulate -> AdamUpdater
          {10,
           8,
           0,
           AdamUpdaterOp::getAccl2InIndex()}, // Accumulate -> AdamUpdater
          {8, 11},                            // AdamUpdater -> AdamVarUpdateOp
          {7, 14},  // ReplicatedReduceScatterOp -> ReplicatedAllReduce
          {14, 12}, // ReplicatedAllReduce -> Scale
          {12, 13}, // Scale -> ScaledAdd
          {13, 9},  // ScaledAdd -> Accumulate
          {13, 10}  // ScaledAdd -> Accumulate
      };

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    case TestOptimizer::Lamb: {
      graphutils::OpPreds preds{
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 0
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 1
          [](const Op *op) { return op->isConvertibleTo<InitOp>(); },       // 2
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 3
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 4
          [](const Op *op) { return op->isConvertibleTo<RemoteLoadOp>(); }, // 5
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllGatherOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          }, // 6
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedReduceScatterOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          }, // 7
          [](const Op *op) {
            return op->isConvertibleTo<AdamUpdaterOp>();
          },                                                                // 8
          [](const Op *op) { return op->isConvertibleTo<AccumulateOp>(); }, // 9
          [](const Op *op) {
            return op->isConvertibleTo<AccumulateOp>();
          }, // 10
          [](const Op *op) {
            return op->isConvertibleTo<AdamVarUpdateOp>();
          },                                                               // 11
          [](const Op *op) { return op->isConvertibleTo<ScaleOp>(); },     // 12
          [](const Op *op) { return op->isConvertibleTo<ScaledAddOp>(); }, // 13
          [](const Op *op) {
            return op->isConvertibleTo<LambSquareOp>();
          }, // 14
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceInplaceOp *>(op);
            return rop && rop->getReplicaGrouping() == aConsecutiveGroup;
          }, // 15
          [&](const Op *op) {
            auto rop = dynamic_cast<const ReplicatedAllReduceOp *>(op);
            return rop && rop->getReplicaGrouping() == anOrthogonalGroup;
          } // 16
      };
      graphutils::Edges edges{
          {0, 3},  // Init -> RemoteLoad
          {1, 4},  // Init -> RemoteLoad
          {2, 5},  // Init -> RemoteLoad
          {3, 11}, // RemoteLoad -> AdamVarUpdateOp
          {3, 6},  // RemoteLoad -> ReplicatedAllGatherOp
          {4, 9},  // RemoteLoad -> AccumulateOp
          {5, 10}, // RemoteLoad -> AccumulateOp
          {9,
           8,
           0,
           AdamUpdaterOp::getAccl1InIndex()}, // Accumulate -> AdamUpdater
          {10,
           8,
           0,
           AdamUpdaterOp::getAccl2InIndex()}, // Accumulate -> AdamUpdater
          {8, 11},                            // AdamUpdater -> AdamVarUpdateOp
          {7, 16},  // ReplicatedReduceScatterOp -> ReplicatedAllReduce
          {16, 12}, // ReplicatedAllReduce -> Scale
          {12, 13}, // Scale -> ScaledAdd
          {13, 9},  // ScaledAdd -> Accumulate
          {13, 10}, // ScaledAdd -> Accumulate
          {8, 14},  // AdamUpdater -> LambSquare
          {14, 15}, // LambSquare -> ReplicatedAllReduceInplace
          {15, 11}, // ReplicatedAllReduceInplace -> AdamVarUpdate
      };

      auto matches = graphutils::findMatchingOps(graph, preds, edges);

      // Expect two instances of the remote exchange / collectives / optimizer
      // pattern (one per weight tensor)
      BOOST_REQUIRE_EQUAL(matches.size(), 2);
      break;
    };
    default: {
      throw error("Optimizer not supported by test");
      break;
    };
    }
  }
}
