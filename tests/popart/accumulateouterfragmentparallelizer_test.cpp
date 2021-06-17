// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AccumulateOuterFragmentParallelizerTest

#include <boost/test/unit_test.hpp>

#include <test_runner.hpp>

#include <popart/names.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/transforms/accumulateouterfragmentparallelizer.hpp>
#include <popart/transforms/transform.hpp>

using namespace popart;

namespace {

using OpGroup  = std::vector<Op *>;
using OpGroups = std::vector<OpGroup>;

void runTest(AccumulateOuterFragmentSettings settings,
             std::function<void(Builder &, TensorId &)> build,
             std::function<void(const std::vector<Op *> &ops)> checker) {

  // Pipeline with two IPUs and two replicas. We should end up with
  // 8 independent weight updates in the accumulate outer fragment
  // of which the stores and loads should be matched accross IPUs
  // in descending order of size.

  TestRunner runner;
  runner.isTraining = true;

  runner.buildModel([&](Builder &builder) {
    auto aiGraphcore = builder.aiGraphcoreOpset1();

    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{2, 10}};
    auto d0 = builder.addInputTensor(inInfo, "data0");
    // builder.virtualGraph(d0, 0);
    auto x = d0;

    build(builder, x);

    auto loss = aiGraphcore.l1loss({x}, 0.1, ReductionType::Mean, "loss");
    builder.virtualGraph(loss, 1);

    runner.anchors.emplace(getGradId(d0), AnchorReturnType("All"));
    runner.opts.enableOutlining                 = false;
    runner.opts.accumulateOuterFragmentSettings = settings;
    runner.opts.outlineThreshold                = 10.0f;
    runner.opts.enableGradientAccumulation      = true;
    runner.opts.accumulationFactor              = 4;
    runner.opts.enableReplicatedGraphs          = true;
    runner.opts.replicatedGraphCount            = 2;
    runner.opts.virtualGraphMode                = VirtualGraphMode::Manual;
    runner.opts.enablePipelining                = true;
    runner.opts.autoRecomputation               = RecomputationType::Pipeline;
    runner.opts.optimizerStateTensorLocationSettings =
        TensorLocationSettings(TensorLocation(TensorStorage::OffChip,
                                              TileSet::Compute,
                                              TileSet::Compute,
                                              ReplicatedTensorSharding::Off),
                               0,
                               0);
    runner.opts.accumulatorTensorLocationSettings =
        TensorLocationSettings(TensorLocation(TensorStorage::OffChip,
                                              TileSet::Compute,
                                              TileSet::Compute,
                                              ReplicatedTensorSharding::Off),
                               0,
                               0);

    runner.patterns = Patterns(PatternsLevel::Default);
    runner.loss     = loss;
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "4"}};
    runner.deviceInfo =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    return x;
  });

  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> ops = ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    checker(ops);
  });
};

// Group ops that match predicate pred into those that are adjacent.
OpGroups getAdjacentOps(const std::vector<Op *> &ops,
                        std::function<bool(Op *)> pred) {
  OpGroups result;
  std::vector<Op *> chunk;
  for (auto op : ops) {
    if (op->settings.executionContext ==
        ExecutionContext::AccumulateOuterFragment) {
      if (pred(op)) {
        chunk.push_back(op);
      } else {
        if (chunk.size() > 0) {
          result.push_back(chunk);
          chunk.clear();
        }
      }
    }
  }
  if (chunk.size() > 0) {
    result.push_back(chunk);
  }
  return result;
};

// Group ops that match predicate pred into those that are adjacent.
std::vector<Op *> filterOps(const std::vector<Op *> &ops,
                            std::function<bool(Op *)> pred) {
  std::vector<Op *> result;
  for (auto op : ops) {
    if (op->settings.executionContext ==
        ExecutionContext::AccumulateOuterFragment) {
      if (pred(op)) {
        result.push_back(op);
      }
    }
  }
  return result;
};

void addMatMul(Builder &builder,
               TensorId &x,
               int64_t rows,
               int64_t cols,
               int vid,
               const std::string &weightId) {
  auto aiOnnx = builder.aiOnnxOpset9();

  TensorInfo weightInfo{"FLOAT", std::vector<int64_t>{rows, cols}};
  std::vector<float> weightDataVec(weightInfo.nelms(), 0);
  ConstVoidData weightData{weightDataVec.data(), weightInfo};

  auto w = builder.addInitializedInputTensor(weightData, weightId);
  // builder.virtualGraph(w, vid);
  x = aiOnnx.matmul({x, w});
  builder.virtualGraph(x, vid);
};

void checkTensor(Op *op,
                 size_t index,
                 const std::string &weightName,
                 const bool isInput) {
  auto tensorIdMap =
      isInput ? op->input->tensorIdMap() : op->output->tensorIdMap();
  const auto &tid = tensorIdMap[index]; // indexMap->tensors()[index]->id;
  const auto msg  = logging::format(
      "expected {} tensor {} ({}) to include string '{}' (for op {})",
      isInput ? "input" : "output",
      index,
      tid,
      weightName,
      op->debugName());
  BOOST_CHECK_MESSAGE(tid.find(weightName) != std::string::npos, msg);
};

void checkMultiExchangeOp(Op *op,
                          std::vector<std::string> loads,
                          std::vector<std::string> stores) {
  // Both loads and stores have two inputs each, the tensor to write to and
  // the RemoteArg___ tensor that contains an offset.
  BOOST_CHECK((loads.size() + stores.size()) * 2 ==
              op->input->tensors().size());

  size_t inputIndex = 0;
  // Check each input has names in the tensor IDs we expect.
  for (auto load : loads) {
    checkTensor(op, inputIndex++, load, true);
    checkTensor(op, inputIndex++, load, true);
  }
  for (auto store : stores) {
    checkTensor(op, inputIndex++, store, true);
    checkTensor(op, inputIndex++, store, true);
  }

  // Outputs only for loads.
  BOOST_CHECK(loads.size() == op->output->tensors().size());

  size_t outputIndex = 0;
  // Check each input has names in the tensor IDs we expect.
  for (auto load : loads) {
    checkTensor(op, outputIndex++, load, false);
  }
};

} // anonymous namespace

BOOST_AUTO_TEST_CASE(
    AccumulateOuterFragmentParallelizer_OverlapCycleOptimized) {

  // Test what happens if we're running OverlapCycleOptimized (weight sizes
  // don't match between VGIDs). To get the best time performance it should
  // combine similarly-matched weights in different virtual graphs into
  // pairwise MultiExchangeOps. Note this should be done in ascending order
  // of weight tensor size to be as gentle on live memory as we can.

  auto config = AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule::OverlapCycleOptimized, {});

  auto build = [&](Builder &builder, TensorId &x) {
    addMatMul(builder, x, 10, 40, 0, "weight0"); // VGID:0, Size: 400
    addMatMul(builder, x, 40, 20, 0, "weight1"); // VGID:0, Size: 800
    addMatMul(builder, x, 20, 25, 0, "weight2"); // VGID:0, Size: 500
    addMatMul(builder, x, 25, 10, 0, "weight3"); // VGID:0, Size: 250
    addMatMul(builder, x, 10, 40, 1, "weight0"); // VGID:1, Size: 400
    addMatMul(builder,
              x,
              40,
              21,
              1,
              "weight1"); // VGID:1, Size: 840 // slight variation in size
                          // compared to VGID 0.
    addMatMul(builder,
              x,
              21,
              25,
              1,
              "weight2"); // VGID:1, Size: 525 // slight variation in size
                          // compared to VGID 0.
    addMatMul(builder, x, 25, 10, 1, "weight3"); // VGID:1, Size: 250
  };

  auto checker = [&](const std::vector<Op *> &ops) {
    auto remoteExOps = filterOps(
        ops, [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });

    // We expect 5 MultiExchangeOps.
    BOOST_CHECK(5 == remoteExOps.size());

    // First op loads 2 x weight3 and stores nothing.
    checkMultiExchangeOp(remoteExOps[0], {"weight3", "weight3"}, {});
    // First op loads 2 x weight0 and stores 2 x weight3.
    checkMultiExchangeOp(
        remoteExOps[1], {"weight0", "weight0"}, {"weight3", "weight3"});
    // First op loads 2 x weight2 and stores 2 x weight0.
    checkMultiExchangeOp(
        remoteExOps[2], {"weight2", "weight2"}, {"weight0", "weight0"});
    // First op loads 2 x weight1 and stores 2 x weight2.
    checkMultiExchangeOp(
        remoteExOps[3], {"weight1", "weight1"}, {"weight2", "weight2"});
    // First op loads nothing and stores 2 x weight1.
    checkMultiExchangeOp(remoteExOps[4], {}, {"weight1", "weight1"});
  };

  runTest(config, build, checker);
}

BOOST_AUTO_TEST_CASE(
    AccumulateOuterFragmentParallelizer_OverlapMemoryOptimized_MatchingWeights) {
  // Test what happens if we're running OverlapMemoryOptimized (weight sizes
  // do match between VGIDs). Note that MultiExchangeOps will not combine
  // RemoteLoadOp and RemoteStoreOps in this mode to give the outliner more
  // opportunity to save memory.

  auto config = AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule::OverlapMemoryOptimized, {});

  auto build = [&](Builder &builder, TensorId &x) {
    addMatMul(builder, x, 10, 40, 0, "weight0"); // VGID:0, Size: 400
    addMatMul(builder, x, 40, 20, 0, "weight1"); // VGID:0, Size: 800
    addMatMul(builder, x, 20, 25, 0, "weight2"); // VGID:0, Size: 500
    addMatMul(builder, x, 25, 10, 0, "weight3"); // VGID:0, Size: 250
    addMatMul(builder, x, 10, 40, 1, "weight0"); // VGID:1, Size: 400
    addMatMul(builder, x, 40, 20, 1, "weight1"); // VGID:1, Size: 800 // same
    addMatMul(builder, x, 20, 25, 1, "weight2"); // VGID:1, Size: 500 // same
    addMatMul(builder, x, 25, 10, 1, "weight3"); // VGID:1, Size: 250
  };

  auto checker = [&](const std::vector<Op *> &ops) {
    auto remoteExOps = filterOps(
        ops, [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });
    // We expect 5 MultiExchangeOps.
    BOOST_CHECK(8 == remoteExOps.size());
    // First op loads 2 x weight3 and stores nothing.
    checkMultiExchangeOp(remoteExOps[0], {"weight3", "weight3"}, {});
    // First op loads 2 x weight0 and, separately, stores 2 x weight3.
    checkMultiExchangeOp(remoteExOps[1], {}, {"weight3", "weight3"});
    checkMultiExchangeOp(remoteExOps[2], {"weight0", "weight0"}, {});
    // First op loads 2 x weight2 and, separately, stores 2 x weight0.
    checkMultiExchangeOp(remoteExOps[3], {}, {"weight0", "weight0"});
    checkMultiExchangeOp(remoteExOps[4], {"weight2", "weight2"}, {});
    // First op loads 2 x weight1 and, separately, stores 2 x weight2.
    checkMultiExchangeOp(remoteExOps[5], {}, {"weight2", "weight2"});
    checkMultiExchangeOp(remoteExOps[6], {"weight1", "weight1"}, {});
    // First op loads nothing and stores 2 x weight1.
    checkMultiExchangeOp(remoteExOps[7], {}, {"weight1", "weight1"});
  };

  runTest(config, build, checker);
}

BOOST_AUTO_TEST_CASE(
    AccumulateOuterFragmentParallelizer_OverlapMemoryOptimized_NonMatchingWeights) {
  // Test what happens if we're running OverlapMemoryOptimized (weight sizes
  // do match between VGIDs). Note that MultiExchangeOps will not combine
  // RemoteLoadOp and RemoteStoreOps in this mode. Also, where weight sizes
  // don't match they will not be combined.

  auto config = AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule::OverlapMemoryOptimized, {});

  auto build = [&](Builder &builder, TensorId &x) {
    addMatMul(builder, x, 10, 40, 0, "weight0"); // VGID:0, Size: 400
    addMatMul(builder, x, 40, 20, 0, "weight1"); // VGID:0, Size: 800
    addMatMul(builder, x, 20, 25, 0, "weight2"); // VGID:0, Size: 500
    addMatMul(builder, x, 25, 10, 0, "weight3"); // VGID:0, Size: 250
    addMatMul(builder, x, 10, 40, 1, "weight0"); // VGID:1, Size: 400
    addMatMul(builder,
              x,
              40,
              21,
              1,
              "weight1"); // VGID:1, Size: 840 // slight variation in size
                          // compared to VGID 0.
    addMatMul(builder,
              x,
              21,
              25,
              1,
              "weight2"); // VGID:1, Size: 525 // slight variation in size
                          // compared to VGID 0.
    addMatMul(builder, x, 25, 10, 1, "weight3"); // VGID:1, Size: 250
  };

  auto checker = [&](const std::vector<Op *> &ops) {
    auto MultiExchangeOps = filterOps(
        ops, [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });

    // We expect 4 MultiExchangeOps.
    BOOST_CHECK(4 == MultiExchangeOps.size());

    // First op loads 2 x weight3.
    checkMultiExchangeOp(MultiExchangeOps[0], {"weight3", "weight3"}, {});
    // Store 2 x weight3.
    checkMultiExchangeOp(MultiExchangeOps[1], {}, {"weight3", "weight3"});
    // Load 2 x weight0.
    checkMultiExchangeOp(MultiExchangeOps[2], {"weight0", "weight0"}, {});
    // Store 2 x weight0.
    checkMultiExchangeOp(MultiExchangeOps[3], {}, {"weight0", "weight0"});
  };

  runTest(config, build, checker);
}

BOOST_AUTO_TEST_CASE(AccumulateOuterFragmentParallelizer_ExcludedVirtualGraph) {
  // Test nothing is parallelised if exclude virtual graph 0.

  auto config = AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule::OverlapMemoryOptimized, {0});

  auto build = [&](Builder &builder, TensorId &x) {
    addMatMul(builder, x, 10, 40, 0, "weight0"); // VGID:0, Size: 400
    addMatMul(builder, x, 40, 20, 0, "weight1"); // VGID:0, Size: 800
    addMatMul(builder, x, 20, 25, 0, "weight2"); // VGID:0, Size: 500
    addMatMul(builder, x, 25, 10, 0, "weight3"); // VGID:0, Size: 250
    addMatMul(builder, x, 10, 40, 1, "weight0"); // VGID:1, Size: 400
    addMatMul(builder, x, 40, 20, 1, "weight1"); // VGID:1, Size: 800
    addMatMul(builder, x, 20, 25, 1, "weight2"); // VGID:1, Size: 500
    addMatMul(builder, x, 25, 10, 1, "weight3"); // VGID:1, Size: 250
  };

  auto checker = [&](const std::vector<Op *> &ops) {
    auto remoteExOps = filterOps(
        ops, [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });

    // We expect no MultiExchangeOps.
    BOOST_CHECK(0 == remoteExOps.size());
  };

  runTest(config, build, checker);
}

BOOST_AUTO_TEST_CASE(
    AccumulateOuterFragmentParallelizer_OverlapMemoryOptimized_NonMatchingShape) {
  // Test that when some weight has the same number of bytes but a different
  // shape and we're in memory optimized mode, we don't introduce remote
  // exchanges.

  auto config = AccumulateOuterFragmentSettings(
      AccumulateOuterFragmentSchedule::OverlapMemoryOptimized, {});

  auto build = [&](Builder &builder, TensorId &x) {
    addMatMul(builder, x, 10, 40, 0, "weight0"); // VGID:0, Size: ...
    addMatMul(builder, x, 40, 20, 0, "weight1"); // VGID:0, Size: 800
    addMatMul(builder, x, 20, 10, 0, "weight2"); // VGID:0, Size: ...
    addMatMul(builder, x, 10, 20, 1, "weight0"); // VGID:1, Size: ...
    addMatMul(
        builder,
        x,
        20,
        40,
        1,
        "weight1"); // VGID:1, Size: Same size as VGID but different shape.
    addMatMul(builder, x, 40, 10, 1, "weight2"); // VGID:1, Size: ...
  };

  auto checker = [&](const std::vector<Op *> &ops) {
    auto remoteExOps = filterOps(
        ops, [](Op *op) { return op->isConvertibleTo<MultiExchangeOp>(); });

    // We expect no MultiExchangeOps.
    BOOST_CHECK(0 == remoteExOps.size());
  };

  runTest(config, build, checker);
}
