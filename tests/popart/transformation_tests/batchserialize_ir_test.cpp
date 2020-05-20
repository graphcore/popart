// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BatchSerializeIrTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/pingpong.hpp>

using namespace popart;

// Model: Repeated N times, batch size M, batch serialized K times:
// ____________________
// IPU 0:
//           in
//            |
// w0 ----- MatMul
//            |
//          ReLU
// ___________|________
// IPU 1:     |
// w1 ----- MatMul
//            |
//          ReLU
//            |
// w2 ----- MatMul
//            |
//          ReLU
// ___________|________
//....
//            |
//           Loss

BOOST_AUTO_TEST_CASE(TestBatchSerialWithVGraphs) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 3;
  int M             = 6;
  int K             = 3;
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

    // 2N layers
    for (int n = 0; n < 2 * N; ++n) {
      // Switch between 1 and 2 MatMul blocks per VGID
      for (int j = 0; j < 1 + n % 2; ++j) {
        auto w = builder.addInitializedInputTensor(wCVData);
        act    = aiOnnx.matmul({act, w},
                            logging::format("CHECKOP_MM: [{} {}]", n, j));
        builder.virtualGraph(act, n % 2);
        act =
            aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{} {}]", n, j));
        builder.virtualGraph(act, n % 2);
      }
    }

    auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
    builder.virtualGraph(loss, (2 * N - 1) % 2);

    runner.opts.batchSerializationFactor = K;
    // Disable outlining (tested separately)
    runner.opts.enableOutlining  = false;
    runner.opts.virtualGraphMode = VirtualGraphMode::Manual;
    runner.patterns              = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.inplaceEnabled = false;
    runner.loss                    = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization:
  // 1.) The schedule repeats the same way for each batch serial
  // 2.) Correct number of total ops with batch serial phases
  // 3.) Correct number of IPU copies (not batch serialized), indicating that
  //     the batch has been concatenated before an IPU copy
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule = ir.getMainGraph().getOpSchedule({});

    size_t numIpuCopies               = 0;
    BatchSerializedPhase currentPhase = -1;
    size_t recordedOffset             = 0;
    std::vector<Op *> recordedOps;

    std::map<BatchSerializedPhase, size_t> opsPerPhase;

    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {} {} {}",
                     op->hasBatchSerializedPhase()
                         ? std::to_string(op->getBatchSerializedPhase())
                         : "*",
                     op->settings.schedulePriority,
                     op->debugName());

      // 3.)
      if (op->isIpuCopyOp()) {
        ++numIpuCopies;
      }

      // 1.)
      if (op->hasBatchSerializedPhase()) {
        opsPerPhase[op->getBatchSerializedPhase()] += 1;

        // Record order of ops for first batch element
        if (op->getBatchSerializedPhase() == 0) {
          if (currentPhase != 0) {
            currentPhase = 0;
            recordedOps.clear();
          }
          recordedOps.push_back(op);
        } else if (op->getBatchSerializedPhase() > 0) {
          if (op->getBatchSerializedPhase() != currentPhase) {
            recordedOffset = 0;
            currentPhase   = op->getBatchSerializedPhase();
          }
          BOOST_CHECK(recordedOps.size() > recordedOffset);
          if (recordedOps.size() > recordedOffset) {
            Op *recordedOp = recordedOps.at(recordedOffset);
            // Identical ops and order of ops per batch serial phase
            BOOST_CHECK(op->opid == recordedOp->opid);
          }
          recordedOffset += 1;
        }
      }
    }

    // 2.)
    for (int k = 0; k < K; ++k) {
      // At least some ops with batch serial phase
      BOOST_CHECK(opsPerPhase[k] > 0);
      // Same number of ops per phase
      BOOST_CHECK(opsPerPhase[k] == opsPerPhase[0]);
    }

    // 3.)
    BOOST_CHECK(numIpuCopies == 4 * N + 1);
  });
}

BOOST_AUTO_TEST_CASE(TestBatchSerialWithVGraphsOutlined) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 2;
  int M             = 8;
  int K             = 4;
  int size          = 100;

  runner.buildModel([&](auto &builder) {
    auto aiOnnx = builder.aiOnnxOpset9();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{M, size}};
    auto act = builder.addInputTensor(inInfo);

    int i = 0;
    // 2N layers
    for (int n = 0; n < 2 * N; ++n) {
      // Switch between 1 and 2 MatMul blocks per VGID
      for (int j = 0; j < 1 + n % 2; ++j) {
        // Decreasing act sizes with every matmul
        // ensures no outlining besides the batch
        // serialzation will happen, to isolate the test condition
        TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size - i, size - i - 1}};
        std::vector<TestTensor> inputs;
        std::vector<TestTensor> outputs;
        std::vector<float> wData(wInfo.nelms(), 0);
        ConstVoidData wCVData{wData.data(), wInfo};
        auto w = builder.addInitializedInputTensor(wCVData);
        act    = aiOnnx.matmul({act, w},
                            logging::format("CHECKOP_MM: [{} {}]", n, j));
        builder.virtualGraph(act, n % 2);
        act =
            aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{} {}]", n, j));
        builder.virtualGraph(act, n % 2);
        ++i;
      }
    }

    auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
    builder.virtualGraph(loss, (2 * N - 1) % 2);

    runner.opts.batchSerializationFactor = K;
    // Enable outlining with no restrictions
    runner.opts.explicitRecomputation          = false;
    runner.opts.enableOutlining                = true;
    runner.opts.outlineThreshold               = -1.0;
    runner.opts.enableOutliningCopyCostPruning = false;
    runner.opts.virtualGraphMode               = VirtualGraphMode::Manual;
    runner.patterns = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.inplaceEnabled = false;
    runner.loss                    = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization:
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule = ir.getMainGraph().getOpSchedule({});

    std::map<GraphId, size_t> numCallsToSubgraph;

    // Testing that the schedule is as expected for batch serialization:
    // 1.) No more batch serialized ops > -1 on the top level graph
    //     after outlining
    // 2.) Expect each subgraph to exist 3 times (once per batch element)
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {} {} {}",
                     op->hasBatchSerializedPhase()
                         ? std::to_string(op->getBatchSerializedPhase())
                         : "*",
                     op->settings.schedulePriority,
                     op->debugName());
      // 1.)
      BOOST_CHECK(!op->hasBatchSerializedPhase() ||
                  op->getBatchSerializedPhase() == -1);

      // 2.)
      for (auto subgraph : op->getCalledGraphs()) {
        ++numCallsToSubgraph[subgraph->id];
      }
    }

    // 2.)
    for (auto &graphIdAndCount : numCallsToSubgraph) {
      // Note due to having two outlining passes, this will vary based on if the
      // second outlining pass can outline or not. In this example, this varies
      // on the chosen batch serialization factor.
      // Expected example values:
      // K = 1: expected = 0
      // K = 2: expected = 2
      //  Call(1), Call(1) -> Call(1), Call(1)
      // K = 3: expected = 3
      //  Call(1), Call(1), Call(1) -> Call(1), Call(1), Call(1)
      // K = 4: expected = 2
      //  Call(1), Call(1), Call(1), Call(1) -> Call(2), Call(2)
      //  Call(2): Call(1), Call(1)
      BOOST_CHECK(graphIdAndCount.second == 2);
    }
  });
}
