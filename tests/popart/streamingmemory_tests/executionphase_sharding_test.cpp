// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExecutionPhaseShardingTest

#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/builder.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/remote.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/transforms/streamingmemory.hpp>

using namespace popart;

// Model: 2x2 S1 ExecutionPhase, repeated N times:
// _____________________________________________________________________________
// phase 0:            IPU 0            |                       IPU 2
// in0 ---- Slice/Slice -----------------------------.
//            |                         |            |
// w0 ----- MatMul                      |          MatMul ----- w1
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//______________________________________X__(inter-phase cross-IPU copy)_________
// phase 1:            IPU 1           /|\                      IPU 3
//            .-----------------------' | '----------.
//            |                         |            |
// w2 ----- MatMul                      |          MatMul ----- w3
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//                                      X  (intra-phase cross-IPU copy)
//                                     /|\
//            .-----------------------' | '----------.
//            |                         |            |
// w4 ----- MatMul                      |          MatMul ----- w5
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//______________________________________X_______________________________________
// phase 2:            IPU 0           /|\                      IPU 2
// ......                               |
// ......                               |
//______________________________________X__(inter-phase cross-IPU copy)_________
// phase N*2-1:        IPU 1           /|\                      IPU 3
//            .-----------------------' | '----------.
//            |                         |            |
// w2 ----- MatMul                      |          MatMul ----- w3
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------.|.-----------+
//                                      X  (intra-phase cross-IPU copy)
//                                     /|\
//            .-----------------------' | '----------.
//            |                         |            |
// w4 ----- MatMul                      |          MatMul ----- w5
//            |                         |            |
//          ReLU                        |           ReLU
//            |                         |            |
//            +------------------------------------ Sum ----- L1Loss
//______________________________________|_______________________________________

BOOST_AUTO_TEST_CASE(Test2x2S1ExecutionPhase) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 5;
  int size          = 100;

  // Weights are [size, size]
  // Input and Acts are [1, size] or [1, 2 * size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx      = builder.aiOnnxOpset9();
    auto aiGraphcore = builder.aiGraphcoreOpset1();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{1, 2 * size}};
    auto input = builder.addInputTensor(inInfo);

    std::vector<std::string> insl0(2);
    std::vector<std::string> insl1(2);
    insl0[0] = aiOnnx.slice({input}, {size}, {0}, {1}, "CHECKOP_SL0");
    insl0[1] = aiOnnx.slice({input}, {2 * size}, {size}, {1}, "CHECKOP_SL1");

    builder.executionPhase(insl0.at(0), 0);
    builder.executionPhase(insl0.at(1), 0);
    builder.virtualGraph(insl0.at(0), 0);
    builder.virtualGraph(insl0.at(1), 0);

    // 2N phases
    for (int n = 0; n < 2 * N; ++n) {
      // Switch between 1 and 2 MatMul blocks per phase
      for (int j = 0; j < 1 + n % 2; ++j) {
        for (int ipu = 0; ipu < 2; ++ipu) {
          auto w        = builder.addInitializedInputTensor(wCVData);
          VGraphId vgid = (n % 2 + 2 * ipu);
          auto out      = aiOnnx.matmul(
              {insl0[ipu], w},
              logging::format("CHECKOP_MM: [{} {}]", n, vgid % 2));
          builder.executionPhase(out, n);
          builder.virtualGraph(out, vgid);
          out = aiOnnx.relu(
              {out}, logging::format("CHECKOP_RELU: [{} {}]", n, vgid % 2));
          builder.executionPhase(out, n);
          builder.virtualGraph(out, vgid);

          // Cross over between IPUs (intra- or inter-phase)
          insl1[(ipu + 1) % 2] = out;
        }
        insl0 = insl1;
      }
    }
    auto sum = aiOnnx.sum(insl0);
    builder.executionPhase(sum, N * 2 - 1);
    builder.virtualGraph(sum, 3);

    auto l1 = aiGraphcore.l1loss({sum}, 0.1);
    builder.executionPhase(l1, N * 2 - 1);
    builder.virtualGraph(l1, 3);

    // To make introspecting the IR easy
    runner.opts.enableOutlining               = false;
    runner.opts.executionPhaseSettings.phases = N * 2;
    runner.opts.executionPhaseSettings.stages = 2;
    runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = l1;

    auto tensorLocation = TensorLocation(TensorStorage::OffChip);

    runner.opts.activationTensorLocationSettings.location     = tensorLocation;
    runner.opts.weightTensorLocationSettings.location         = tensorLocation;
    runner.opts.optimizerStateTensorLocationSettings.location = tensorLocation;
    runner.opts.accumulatorTensorLocationSettings.location    = tensorLocation;

    return sum;
  });

  // Testing that the schedule makes sense for 2x2 ExecutionPhase execution:
  // 1.) VGIDs and ExecutionPhases of the MatMul and ReLU stay consistent
  // 2.) Inter- and intra-phase IpuCopyOps are placed correctly
  // 3.) Initial SliceOps is placed correctly
  // 4.) Final loss op is placed correctly
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {}", op->debugName());

      // 1.)
      if (op->getName().find("CHECKOP_MM") != std::string::npos) {
        ExecutionPhase n = op->getExecutionPhase();
        VGraphId vgid    = op->getVirtualGraphId();
        if (op->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_MM: [{} {}]", n, vgid % 2)) !=
                      std::string::npos);
        }
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_MM: [{} {}]", N * 4 - n - 2, vgid % 2)) !=
                      std::string::npos);
        }
      }
      if (op->getName().find("CHECKOP_RELU") != std::string::npos) {
        ExecutionPhase n = op->getExecutionPhase();
        VGraphId vgid    = op->getVirtualGraphId();
        if (op->toLoss == PathToLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_RELU: [{} {}]", n, vgid % 2)) !=
                      std::string::npos);
        }
        if (op->fromLoss == PathFromLoss::Yes) {
          BOOST_CHECK(op->getName().find(logging::format(
                          "CHECKOP_RELU: [{} {}]", N * 4 - n - 2, vgid % 2)) !=
                      std::string::npos);
        }
      }

      // 2.)
      if (op->isIpuCopyOp()) {
        // IpuCopyOps should not have the VGID set
        BOOST_CHECK(!op->hasVirtualGraphId());
        IpuCopyOp *copy = dynamic_cast<IpuCopyOp *>(op);
        if (copy) {
          if (copy->getSourceIpu() % 2 != copy->getDestIpu() % 2) {
            // Inter-phase
            // See streamingmemoryopinserter.cpp ipuCopyPriority
            BOOST_CHECK(copy->settings.schedulePriority < -9000.0);
          } else {
            // Intra-phase
            BOOST_CHECK(copy->settings.schedulePriority == 0.0);
          }
        }
      }

      // 3.)
      if (op->getName().find("CHECKOP_SL0") != std::string::npos ||
          op->getName().find("CHECKOP_SL1") != std::string::npos) {
        BOOST_CHECK(op->getVirtualGraphId() == 0);
        BOOST_CHECK(op->getExecutionPhase() == 0);
      }

      // 4.)
      if (op->isLossOp()) {
        BOOST_CHECK(op->getVirtualGraphId() == 3);
        BOOST_CHECK(op->getExecutionPhase() == N * 2 - 1);
      }
    }
  });
}

// Model: 1x0 S1 ExecutionPhase, repeated N times
// Keeps activations between adjacent phases (phase stride 1)
// Overlaps IO and compute with adjacent phases
BOOST_AUTO_TEST_CASE(Test1x0S1ExecutionPhase) {
  TestRunner runner;
  runner.isTraining  = true;
  int batchSize      = 8;
  int batchSerialize = 4;
  int N              = 5;
  int size           = 100;

  // Weights are [size, size]
  // Input and Acts are [batchSize, size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx      = builder.aiOnnxOpset9();
    auto aiGraphcore = builder.aiGraphcoreOpset1();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{batchSize, 1, size}};
    auto input = builder.addInputTensor(inInfo);

    // N phases
    for (int n = 0; n < N; ++n) {
      auto w = builder.addInitializedInputTensor(wCVData);
      auto out =
          aiOnnx.matmul({input, w}, logging::format("CHECKOP_MM: [{}]", n));
      builder.executionPhase(out, n);
      builder.virtualGraph(out, 0);
      out = aiOnnx.relu({out}, logging::format("CHECKOP_RELU: [{}]", n));
      builder.executionPhase(out, n);
      builder.virtualGraph(out, 0);

      // Cross over between IPUs (intra-phase)
      input = out;
    }

    auto l1 = aiGraphcore.l1loss({input}, 0.1);
    builder.executionPhase(l1, N - 1);
    builder.virtualGraph(l1, 0);

    // To make introspecting the IR easy
    runner.opts.enableOutlining                   = false;
    runner.opts.autoRecomputation                 = RecomputationType::Standard;
    runner.opts.explicitRecomputation             = true;
    runner.opts.batchSerializationSettings.factor = batchSerialize;
    runner.opts.executionPhaseSettings.phases     = N;
    runner.opts.executionPhaseSettings.stages     = 1;
    runner.opts.executionPhaseSettings.activationIOSchedule =
        ExecutionPhaseIOSchedule::Preload;
    runner.opts.executionPhaseSettings.weightIOSchedule =
        ExecutionPhaseIOSchedule::Preload;
    runner.opts.executionPhaseSettings.optimizerStateIOSchedule =
        ExecutionPhaseIOSchedule::Preload;
    runner.opts.executionPhaseSettings.accumulatorIOSchedule =
        ExecutionPhaseIOSchedule::Preload;
    runner.opts.executionPhaseSettings.schedule = ExecutionPhaseSchedule::Batch;
    runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = l1;
    runner.opts.numIOTiles       = 192;

    auto tensorLocation = TensorLocation(TensorStorage::OffChip,
                                         TileSet::IO,
                                         TileSet::IO,
                                         ReplicatedTensorSharding::Off);

    runner.opts.activationTensorLocationSettings.location     = tensorLocation;
    runner.opts.weightTensorLocationSettings.location         = tensorLocation;
    runner.opts.optimizerStateTensorLocationSettings.location = tensorLocation;
    runner.opts.accumulatorTensorLocationSettings.location    = tensorLocation;
    return input;
  });

  runner.checkIr([&](Ir &ir) {
    std::map<ExecutionPhase, std::tuple<int, int, int, int>> remoteOpsPerPhase;

    std::set<ExecutionPhase> ioTileCopyInPhase;
    std::set<ExecutionPhase> remoteStoreInPhase;

    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace(
          "Op: {} {}", op->debugName(), op->settings.schedulePriority);

      ExecutionPhase phase = op->getExecutionPhase();
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<0>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<1>(remoteOpsPerPhase[phase]) += 1;
        }
      }

      if (op->isConvertibleTo<RemoteStoreOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<2>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<3>(remoteOpsPerPhase[phase]) += 1;
        }
      }

      // Check strict ordering in each phase due to having
      // ExecutionPhaseSchedule::Batch
      // enabled: RemoteLoad before IOTileCopy before RemoteStore
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        BOOST_CHECK(ioTileCopyInPhase.find(op->getExecutionPhase()) ==
                    ioTileCopyInPhase.end());
        BOOST_CHECK(remoteStoreInPhase.find(op->getExecutionPhase()) ==
                    remoteStoreInPhase.end());
      }
      if (op->isConvertibleTo<IoTileCopyOp>()) {
        BOOST_CHECK(remoteStoreInPhase.find(op->getExecutionPhase()) ==
                    remoteStoreInPhase.end());
        ioTileCopyInPhase.insert(op->getExecutionPhase());
      }
      if (op->isConvertibleTo<RemoteStoreOp>()) {
        remoteStoreInPhase.insert(op->getExecutionPhase());
      }
    }

    // For this execution mode, we expect:
    // 1.) For phases -1, 0, ..., N - 1: Preload one weight, #N in total
    // 2.) For phases 1, ..., N - 2: Save one activation, #N - 3 in total
    // 3.) For phases N - 1, ..., 2 * N - 2: Save one weight, #N in total
    // 4.) For phases N, ..., 2 * N - 4: Load a weight and an activation
    // 4.) For phases 2 * N - 3: Load a weight
    for (auto &kv : remoteOpsPerPhase) {
      logging::trace("Remote ops in phase: {} {} {} {} {}",
                     kv.first,
                     std::get<0>(kv.second),
                     std::get<1>(kv.second),
                     std::get<2>(kv.second),
                     std::get<3>(kv.second));
      // 1.)
      if (kv.first < N - 1) {
        BOOST_CHECK(std::get<1>(kv.second) == 1);
      }
      // 2.)
      if (kv.first < N - 2 && kv.first > 0) {
        BOOST_CHECK(std::get<3>(kv.second) == 1);
      }
      // 3.)
      if (kv.first >= N - 1) {
        BOOST_CHECK(std::get<3>(kv.second) == 1);
      }
      // 4.)
      if (kv.first >= N && kv.first < 2 * N - 3) {
        BOOST_CHECK(std::get<3>(kv.second) == 1);
      }
      // 4.)
      if (kv.first == 2 * N - 3) {
        BOOST_CHECK(std::get<3>(kv.second) == 1);
      }
    }
  });
}

// Model: 1x0 S2 ExecutionPhase, repeated N times
// Keeps activations between adjacent phases (phase stride 2)
BOOST_AUTO_TEST_CASE(Test1x0S2ExecutionPhase) {
  TestRunner runner;
  runner.isTraining  = true;
  int batchSize      = 8;
  int batchSerialize = 4;
  int N              = 5;
  int size           = 100;

  // Weights are [size, size]
  // Input and Acts are [batchSize, size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx      = builder.aiOnnxOpset9();
    auto aiGraphcore = builder.aiGraphcoreOpset1();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{batchSize, size}};
    auto input = builder.addInputTensor(inInfo);

    // N phases
    for (int n = 0; n < N; ++n) {
      auto w = builder.addInitializedInputTensor(wCVData);
      auto out =
          aiOnnx.matmul({input, w}, logging::format("CHECKOP_MM: [{}]", n));
      builder.executionPhase(out, n * 2);
      builder.virtualGraph(out, 0);
      out = aiOnnx.relu({out}, logging::format("CHECKOP_RELU: [{}]", n));
      builder.executionPhase(out, n * 2);
      builder.virtualGraph(out, 0);

      // Cross over between IPUs (intra-phase)
      input = out;
    }

    auto l1 = aiGraphcore.l1loss({input}, 0.1);
    builder.executionPhase(l1, N * 2 - 2);
    builder.virtualGraph(l1, 0);

    // To make introspecting the IR easy
    runner.opts.enableOutlining                   = false;
    runner.opts.batchSerializationSettings.factor = batchSerialize;
    runner.opts.batchSerializationSettings.concatOnVirtualGraphChange   = false;
    runner.opts.batchSerializationSettings.concatOnExecutionPhaseChange = false;
    runner.opts.executionPhaseSettings.phases = N * 2 - 1;
    runner.opts.executionPhaseSettings.stages = 1;
    runner.opts.executionPhaseSettings.activationIOSchedule =
        ExecutionPhaseIOSchedule::OnDemand;
    runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = l1;

    auto tensorLocation = TensorLocation(TensorStorage::OffChip);

    runner.opts.activationTensorLocationSettings.location     = tensorLocation;
    runner.opts.weightTensorLocationSettings.location         = tensorLocation;
    runner.opts.optimizerStateTensorLocationSettings.location = tensorLocation;
    runner.opts.accumulatorTensorLocationSettings.location    = tensorLocation;

    return input;
  });

  runner.checkIr([&](Ir &ir) {
    std::map<ExecutionPhase, std::tuple<int, int, int, int>> remoteOpsPerPhase;

    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {}", op->debugName());

      ExecutionPhase phase = op->getExecutionPhase();
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<0>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<1>(remoteOpsPerPhase[phase]) += 1;
        }
      }

      if (op->isConvertibleTo<RemoteStoreOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<2>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<3>(remoteOpsPerPhase[phase]) += 1;
        }
      }
    }

    // For this execution mode, we expect:
    // 1.) For phases -1, 1, 3, ..., 7: Preload one weight, #N in total
    // 2.) For phases 0, 2, 4, 6: Store #batchSerialize activations on-demand
    // 3.) For phases 10, 12, 14, 16: Load #batchSerialize activations on-demand
    // 4.) For phases 8, 10, 12, 14, 16: Store one weight, #N in total
    // 5.) For phases 9, 11, 13, 15: Preload one weight, #N - 1 in total
    //     (the weight load in phase 7 is shared between fwd and bwd pass)
    for (auto &kv : remoteOpsPerPhase) {
      logging::trace("Remote ops in phase: {} {} {} {} {}",
                     kv.first,
                     std::get<0>(kv.second),
                     std::get<1>(kv.second),
                     std::get<2>(kv.second),
                     std::get<3>(kv.second));
      // 1.)
      if (kv.first < N * 2 - 2 && kv.first % 2 == 1) {
        BOOST_CHECK(std::get<1>(kv.second) == 1);
      }
      // 2.)
      if (kv.first < N * 2 - 2 && kv.first % 2 == 0) {
        BOOST_CHECK(std::get<2>(kv.second) == batchSerialize);
      }
      // 3.)
      if (kv.first > N * 2 - 2 && kv.first % 2 == 0) {
        BOOST_CHECK(std::get<0>(kv.second) == batchSerialize);
      }
      // 4.)
      if (kv.first >= N * 2 - 2 && kv.first % 2 == 0) {
        BOOST_CHECK(std::get<3>(kv.second) == 1);
      }
      // 5.)
      if (kv.first > N * 2 - 2 && kv.first % 2 == 1) {
        BOOST_CHECK(std::get<1>(kv.second) == 1);
      }
    }
  });
}

// Model: 1x0 S4 ExecutionPhase, repeated N times
// Stores and loads activations between all phases (phase stride 4)
BOOST_AUTO_TEST_CASE(Test1x0S4ExecutionPhase) {
  auto run_test = [](bool activationsThroughIOTiles) {
    TestRunner runner;
    runner.isTraining  = true;
    int batchSize      = 8;
    int batchSerialize = 4;
    int N              = 5;
    int size           = 100;

    // Weights are [size, size]
    // Input and Acts are [batchSize, size]
    TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
    std::vector<TestTensor> inputs;
    std::vector<TestTensor> outputs;
    std::vector<float> wData(wInfo.nelms(), 0);
    ConstVoidData wCVData{wData.data(), wInfo};

    runner.buildModel([&](auto &builder) {
      auto aiOnnx      = builder.aiOnnxOpset9();
      auto aiGraphcore = builder.aiGraphcoreOpset1();
      TensorInfo inInfo{"FLOAT", std::vector<int64_t>{batchSize, size}};
      auto input = builder.addInputTensor(inInfo);

      // N phases
      for (int n = 0; n < N; ++n) {
        auto w = builder.addInitializedInputTensor(wCVData);
        auto out =
            aiOnnx.matmul({input, w}, logging::format("CHECKOP_MM: [{}]", n));
        builder.executionPhase(out, n * 4);
        builder.virtualGraph(out, 0);
        out = aiOnnx.relu({out}, logging::format("CHECKOP_RELU: [{}]", n));
        builder.executionPhase(out, n * 4);
        builder.virtualGraph(out, 0);

        // Cross over between IPUs (intra-phase)
        input = out;
      }

      auto l1 = aiGraphcore.l1loss({input}, 0.1);
      builder.executionPhase(l1, N * 4 - 4);
      builder.virtualGraph(l1, 0);

      // To make introspecting the IR easy
      runner.opts.enableOutlining                   = false;
      runner.opts.batchSerializationSettings.factor = batchSerialize;
      runner.opts.batchSerializationSettings.concatOnVirtualGraphChange = false;
      runner.opts.batchSerializationSettings.concatOnExecutionPhaseChange =
          false;
      runner.opts.executionPhaseSettings.phases = N * 4 - 3;
      runner.opts.executionPhaseSettings.stages = 1;
      runner.opts.executionPhaseSettings.activationIOSchedule =
          ExecutionPhaseIOSchedule::OnDemand;
      runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
      runner.patterns              = Patterns(PatternsLevel::Default);
      runner.loss                  = l1;

      auto tensorLocation = TensorLocation(TensorStorage::OffChip);

      runner.opts.activationTensorLocationSettings.location = tensorLocation;
      runner.opts.weightTensorLocationSettings.location     = tensorLocation;
      runner.opts.optimizerStateTensorLocationSettings.location =
          tensorLocation;
      runner.opts.accumulatorTensorLocationSettings.location = tensorLocation;

      if (activationsThroughIOTiles) {
        runner.opts.activationTensorLocationSettings.location.loadTileSet =
            TileSet::IO;
        runner.opts.numIOTiles = 192;
      }

      return input;
    });

    runner.checkIr([&](Ir &ir) {
      std::map<ExecutionPhase, std::tuple<int, int, int, int>>
          remoteOpsPerPhase;
      std::map<ExecutionPhase, int> ioTileCopyOpsPerPhase;

      std::vector<Op *> schedule =
          ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
      for (size_t i = 0; i < schedule.size(); i++) {
        Op *op = schedule.at(i);
        logging::trace("Op: {}", op->debugName());

        ExecutionPhase phase = op->getExecutionPhase();
        if (op->isConvertibleTo<RemoteLoadOp>()) {
          if (op->settings.schedulePriority == 0.0) {
            std::get<0>(remoteOpsPerPhase[phase]) += 1;
          } else {
            std::get<1>(remoteOpsPerPhase[phase]) += 1;
          }
        }

        if (op->isConvertibleTo<RemoteStoreOp>()) {
          if (op->settings.schedulePriority == 0.0) {
            std::get<2>(remoteOpsPerPhase[phase]) += 1;
          } else {
            std::get<3>(remoteOpsPerPhase[phase]) += 1;
          }
        }

        if (op->isConvertibleTo<IoTileCopyOp>()) {
          ioTileCopyOpsPerPhase[phase] += 1;
        }
      }

      // For this execution mode, we expect:
      // 1.) For phases -1, 3, 7, ..., 31: Preload one weight
      // 2.) For phases 0: Store #batchSerialize * 2 activations on-demand
      //     (the output activation and the network input)
      // 3.) For phases 4, 8, ..., 28: Store #batchSerialize activations
      // on-demand 4.) For phases 4, 8, ..., 16: Load #batchSerialize
      // activations on-demand 5.) For phases 20, 24 ..., 32: Load 3 *
      // #batchSerialize activations on-demand
      //     (fwd activation * 2 + gradient)
      // 6.) For phases 16, 20, 24, 28, 32: Store one weight, #N in total
      // 7.) If activationsThroughIOTiles: In each phase, number of IO tile copy
      //     ops corresponds to the number of activations stored + loaded
      //     in that phase
      for (auto &kv : remoteOpsPerPhase) {
        auto ioTileOps = ioTileCopyOpsPerPhase.find(kv.first);
        auto numIoTileOps =
            ioTileOps == ioTileCopyOpsPerPhase.end() ? 0 : ioTileOps->second;

        logging::trace("Remote ops in phase: {} {} {} {} {} {}",
                       kv.first,
                       std::get<0>(kv.second),
                       std::get<1>(kv.second),
                       std::get<2>(kv.second),
                       std::get<3>(kv.second),
                       numIoTileOps);
        // 1.)
        if (kv.first == -1 || (kv.first < N * 8 - 8 && kv.first % 4 == 1)) {
          BOOST_CHECK(std::get<1>(kv.second) == 1);
        }
        // 2.)
        if (kv.first == 0) {
          BOOST_CHECK(std::get<2>(kv.second) == 2 * batchSerialize);
        }
        // 3.)
        if (kv.first > 0 && kv.first < N * 8 - 8 && kv.first % 4 == 0) {
          BOOST_CHECK(std::get<2>(kv.second) == batchSerialize);
        }
        // 4.)
        if (kv.first > 0 && kv.first < N * 4 && kv.first % 4 == 0) {
          BOOST_CHECK(std::get<0>(kv.second) == batchSerialize);
        }
        // 5.)
        if (kv.first >= N * 4 && kv.first % 4 == 0) {
          BOOST_CHECK(std::get<0>(kv.second) == 3 * batchSerialize);
        }
        // 6.)
        if (kv.first >= N * 4 - 4 && kv.first % 4 == 0) {
          BOOST_CHECK(std::get<3>(kv.second) == 1);
        }
        // 7.)
        if (activationsThroughIOTiles) {
          BOOST_CHECK(numIoTileOps ==
                      std::get<0>(kv.second) + std::get<2>(kv.second));
        } else {
          BOOST_CHECK(numIoTileOps == 0);
        }
      }
    });
  };
  run_test(false);
  run_test(true);
}

// Model: 2x0 S2 ExecutionPhase, repeated N times
BOOST_AUTO_TEST_CASE(Test2x0S2ExecutionPhase) {
  TestRunner runner;
  runner.isTraining  = true;
  int batchSize      = 8;
  int batchSerialize = 4;
  int N              = 5;
  int size           = 100;

  // Weights are [size, size]
  // Input and Acts are [batchSize, size] or [batchSize, 2 * size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{1, size, size}};
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  runner.buildModel([&](auto &builder) {
    auto aiOnnx      = builder.aiOnnxOpset9();
    auto aiGraphcore = builder.aiGraphcoreOpset1();
    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{batchSize, 2 * size}};
    auto input = builder.addInputTensor(inInfo);

    std::vector<std::string> insl0(2);
    std::vector<std::string> insl1(2);
    insl0[0] = aiOnnx.slice({input}, {size}, {0}, {1}, "CHECKOP_SL0");
    insl0[1] = aiOnnx.slice({input}, {2 * size}, {size}, {1}, "CHECKOP_SL1");

    builder.executionPhase(insl0.at(0), 0);
    builder.executionPhase(insl0.at(1), 0);
    builder.virtualGraph(insl0.at(0), 0);
    builder.virtualGraph(insl0.at(1), 0);

    // 2N phases
    for (int n = 0; n < 2 * N; ++n) {
      // Switch between 1 and 2 MatMul blocks per phase
      for (int j = 0; j < 1 + n % 2; ++j) {
        for (int vgid = 0; vgid < 2; ++vgid) {
          auto w   = builder.addInitializedInputTensor(wCVData);
          auto out = aiOnnx.matmul(
              {insl0[vgid], w},
              logging::format("CHECKOP_MM: [{} {}]", n, vgid % 2));
          builder.executionPhase(out, n * 2);
          builder.virtualGraph(out, vgid);
          out = aiOnnx.relu(
              {out}, logging::format("CHECKOP_RELU: [{} {}]", n, vgid % 2));
          builder.executionPhase(out, n * 2);
          builder.virtualGraph(out, vgid);

          // Cross over between IPUs (intra-phase)
          insl1[(vgid + 1) % 2] = out;
        }
        insl0 = insl1;
      }
    }
    auto sum = aiOnnx.sum(insl0);
    builder.executionPhase(sum, N * 4 - 2);
    builder.virtualGraph(sum, 0);

    auto l1 = aiGraphcore.l1loss({sum}, 0.1);
    builder.executionPhase(l1, N * 4 - 2);
    builder.virtualGraph(l1, 0);

    // To make introspecting the IR easy
    runner.opts.enableOutlining                   = false;
    runner.opts.batchSerializationSettings.factor = batchSerialize;
    runner.opts.batchSerializationSettings.concatOnVirtualGraphChange   = false;
    runner.opts.batchSerializationSettings.concatOnExecutionPhaseChange = false;
    runner.opts.executionPhaseSettings.phases = N * 4 - 1;
    runner.opts.executionPhaseSettings.stages = 1;
    runner.opts.executionPhaseSettings.activationIOSchedule =
        ExecutionPhaseIOSchedule::OnDemand;
    runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
    runner.patterns              = Patterns(PatternsLevel::Default);
    runner.loss                  = l1;

    auto tensorLocation = TensorLocation(TensorStorage::OffChip);

    runner.opts.activationTensorLocationSettings.location     = tensorLocation;
    runner.opts.weightTensorLocationSettings.location         = tensorLocation;
    runner.opts.optimizerStateTensorLocationSettings.location = tensorLocation;
    runner.opts.accumulatorTensorLocationSettings.location    = tensorLocation;

    return sum;
  });

  runner.checkIr([&](Ir &ir) {
    std::map<ExecutionPhase, std::tuple<int, int, int, int>> remoteOpsPerPhase;

    std::vector<Op *> schedule =
        ir.getOpSchedule({}, RequireOptimalSchedule::Yes);
    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace(
          "Op: {} {}", op->debugName(), op->settings.schedulePriority);

      ExecutionPhase phase = op->getExecutionPhase();
      if (op->isConvertibleTo<RemoteLoadOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<0>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<1>(remoteOpsPerPhase[phase]) += 1;
        }
      }

      if (op->isConvertibleTo<RemoteStoreOp>()) {
        if (op->settings.schedulePriority == 0.0) {
          std::get<2>(remoteOpsPerPhase[phase]) += 1;
        } else {
          std::get<3>(remoteOpsPerPhase[phase]) += 1;
        }
      }
    }

    // For this execution mode, we expect:
    // 1.) For phases -1, 3, 7, ..., 35: Preload 2 weights
    // 2.) For phases 1, 5, 9, ..., 33: Preload 4 weights
    // 3.) For phases 20, ..., 36: Store 2 weights
    // 4.) For phases 18, ..., 34: Store 4 weights
    // 5.) For phases 0, ..., 16: Store 16 activations
    // 6.) For phases 2, ..., 14: Store 32 activations
    // 7.) For phases 20, ..., 36: Load 16 activations
    // 8.) For phases 22, ..., 34: Load 32 activations
    for (auto &kv : remoteOpsPerPhase) {
      logging::trace("Remote ops in phase: {} {} {} {} {}",
                     kv.first,
                     std::get<0>(kv.second),
                     std::get<1>(kv.second),
                     std::get<2>(kv.second),
                     std::get<3>(kv.second));
      // 1.)
      if (kv.first < N * 8 - 4 && kv.first % 4 == 3) {
        BOOST_CHECK(std::get<1>(kv.second) == 2);
      }
      // 2.)
      if (kv.first < N * 8 - 4 && kv.first % 4 == 1) {
        BOOST_CHECK(std::get<1>(kv.second) == 4);
      }
      // 3.) & 7.)
      if (kv.first > N * 4 - 4 && kv.first % 4 == 0) {
        BOOST_CHECK(std::get<3>(kv.second) == 2);
        BOOST_CHECK(std::get<0>(kv.second) == 16);
      }
      // 4.) & 8.)
      if (kv.first > N * 4 - 4 && kv.first % 4 == 2) {
        BOOST_CHECK(std::get<3>(kv.second) == 4);
        if (kv.first > N * 4) {
          // Because last fwd & first bwd are adjacent
          BOOST_CHECK(std::get<0>(kv.second) == 32);
        }
      }
      // 5.)
      if (kv.first <= N * 4 - 4 && kv.first % 4 == 0) {
        BOOST_CHECK(std::get<2>(kv.second) == 16);
      }
      // 6.)
      if (kv.first <= N * 4 - 4 && kv.first % 4 == 2) {
        BOOST_CHECK(std::get<2>(kv.second) == 32);
      }
    }
  });
}
