// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BatchSerializeIrTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Hack to allow the test to view the private data of classes
#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif
#define private public
#define protected public

#include <popart/adam.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "../test_runner.hpp"
#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/optimizer.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/voiddata.hpp"

using namespace popart;

// Model:
// ____________________
// PS 0:
//           in
//            |
// w0 ----- MatMul
//            |
//          ReLU
// ___________|________
// PS 1:      |
// w1 ----- MatMul
//            |
//          ReLU
//            |
//           Loss

BOOST_AUTO_TEST_CASE(TestAccumulatorInplace) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 2;
  int M             = 6;
  int size          = 100;

  // Weights are [size, size]
  // Input and Acts are [M, size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{M, size}};
    auto act = builder.addInputTensor(inInfo);

    // N layers
    for (int n = 0; n < N; ++n) {
      auto w = builder.addInitializedInputTensor(wCVData);
      act    = aiOnnx.matmul({act, w}, logging::format("CHECKOP_MM: [{}]", n));
      builder.virtualGraph(act, n);
      builder.pipelineStage(act, n);
      act = aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{}]", n));
      builder.virtualGraph(act, n);
      builder.pipelineStage(act, n);
    }

    auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
    builder.virtualGraph(loss, N - 1);
    builder.pipelineStage(loss, N - 1);

    // Disable outlining (tested separately)
    runner.deviceInfo                      = createTestDevice(TEST_TARGET, 4);
    runner.opts.enableOutlining            = false;
    runner.opts.virtualGraphMode           = VirtualGraphMode::Manual;
    runner.opts.enableReplicatedGraphs     = true;
    runner.opts.replicatedGraphCount       = 2;
    runner.opts.enableGradientAccumulation = true;
    runner.opts.accumulationFactor         = 3;
    runner.opts.enablePipelining           = true;
    runner.opts.autoRecomputation          = RecomputationType::Pipeline;
    std::map<std::string, std::pair<float, bool>> optValues;
    runner.optimizer               = std::make_unique<Adam>(optValues,
                                              AdamMode::Lamb,
                                              WeightDecayMode::Decay,
                                              DataType::FLOAT,
                                              DataType::FLOAT,
                                              DataType::FLOAT);
    runner.patterns                = Patterns(PatternsLevel::Default);
    runner.patterns.inplaceEnabled = true;
    runner.loss                    = loss;

    return act;
  });

  // Testing that the accumulator reduction is inplaced
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::No);

    std::vector<Op *> recordedOps;

    auto numAllReduce        = 0;
    auto numAllReduceInplace = 0;

    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {} {} {}",
                     op->hasBatchSerializedPhase()
                         ? std::to_string(op->getBatchSerializedPhase())
                         : "*",
                     op->settings.schedulePriority,
                     op->debugName());

      if (op->isConvertibleTo<ReplicatedAllReduceOp>()) {
        ++numAllReduce;
      }
      if (op->isConvertibleTo<ReplicatedAllReduceInplaceOp>()) {
        ++numAllReduceInplace;
      }
    }

    logging::trace(
        "AllReduce: {}, inplace: {}", numAllReduce, numAllReduceInplace);
    BOOST_CHECK(numAllReduce == 2);
    BOOST_CHECK(numAllReduceInplace == 2);
  });
}
