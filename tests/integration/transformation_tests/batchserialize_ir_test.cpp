// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BatchSerializeIrTest

#include "../test_runner.hpp" // IWYU pragma: keep

#include <algorithm>
#include <array>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/exchange/multiexchange.hpp>
#include <popart/op/exchange/remote.hpp>
#include <popart/op/iotilecopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/sum.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/testdevice.hpp>

#include "../random_util.hpp"
#include "popart/builder.gen.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/ndarraywrapper.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/scheduler_requireoptimal.hpp"
#include "popart/sgd.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/voiddata.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

namespace {

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

std::pair<TensorId, TensorId>
buildBatchSerialModel(Builder &builder, int M, int N, int size) {
  // Weights are [size, size]
  // Input and Acts are [M, size]
  TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
  std::vector<float> wData(wInfo.nelms(), 0);
  ConstVoidData wCVData{wData.data(), wInfo};

  auto aiOnnx = builder.aiOnnxOpset9();
  TensorInfo inInfo{"FLOAT", std::vector<int64_t>{M, size}};
  auto act = builder.addInputTensor(inInfo);

  // 2N layers
  for (int n = 0; n < 2 * N; ++n) {
    // Switch between 1 and 2 MatMul blocks per VGID
    for (int j = 0; j < 1 + n % 2; ++j) {
      auto w = builder.addInitializedInputTensor(wCVData);
      act =
          aiOnnx.matmul({act, w}, logging::format("CHECKOP_MM: [{} {}]", n, j));
      builder.virtualGraph(act, n % 2);
      act = aiOnnx.relu({act}, logging::format("CHECKOP_RELU: [{} {}]", n, j));
      builder.virtualGraph(act, n % 2);
    }
  }

  auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
  builder.virtualGraph(loss, (2 * N - 1) % 2);
  return {loss, act};
}

} // namespace

BOOST_AUTO_TEST_CASE(TestBatchSerialWithVGraphs) {
  TestRunner runner;
  runner.isTraining = true;
  int N             = 3;
  int M             = 6;
  int K             = 3;
  int size          = 100;

  runner.buildModel([&](auto &builder) {
    auto lossAct  = buildBatchSerialModel(builder, M, N, size);
    TensorId loss = lossAct.first;
    TensorId act  = lossAct.second;

    runner.opts.batchSerializationSettings.factor = K;
    // Disable outlining (tested separately)
    runner.opts.enableOutlining  = false;
    runner.opts.virtualGraphMode = VirtualGraphMode::Manual;
    runner.patterns              = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.enableInPlace(false);
    runner.loss = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization:
  // 1.) The schedule repeats the same way for each batch serial
  // 2.) Correct number of total ops with batch serial phases
  // 3.) Correct number of IPU copies (not batch serialized), indicating that
  //     the batch has been concatenated before an IPU copy
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

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
      BOOST_CHECK_GT(opsPerPhase[k], 0);
      // Same number of ops per phase
      BOOST_CHECK_EQUAL(opsPerPhase[k], opsPerPhase[0]);
    }

    // 3.)
    BOOST_CHECK_EQUAL(numIpuCopies, 10);
  });
}

BOOST_AUTO_TEST_CASE(TestBatchSerialWithVGraphsBwd) {
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
    auto lossAct  = buildBatchSerialModel(builder, M, N, size);
    TensorId loss = lossAct.first;
    TensorId act  = lossAct.second;

    runner.opts.batchSerializationSettings.factor = K;
    // Test if the graph can be serialized after the backward pass is grown
    runner.opts.batchSerializationSettings.transformContext =
        BatchSerializationTransformContext::Bwd;
    // Disable outlining (tested separately)
    runner.opts.enableOutlining  = false;
    runner.opts.virtualGraphMode = VirtualGraphMode::Manual;
    runner.patterns              = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.enableInPlace(false);
    runner.loss = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization:
  // 1.) The schedule repeats the same way for each batch serial
  // 2.) Correct number of total ops with batch serial phases
  // 3.) Correct number of IPU copies (not batch serialized), indicating that
  //     the batch has been concatenated before an IPU copy
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

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
      BOOST_CHECK_GT(opsPerPhase[k], 0);
      // Same number of ops per phase
      BOOST_CHECK_EQUAL(opsPerPhase[k], opsPerPhase[0]);
    }

    // 3.)
    BOOST_CHECK_EQUAL(numIpuCopies, 4 * N - 2);
  });
}

BOOST_AUTO_TEST_CASE(TestBatchSerialWithVGraphsBwdLoop) {
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
    auto lossAct  = buildBatchSerialModel(builder, M, N, size);
    TensorId loss = lossAct.first;
    TensorId act  = lossAct.second;

    runner.opts.batchSerializationSettings.factor = K;
    // Test if the graph can be serialized after the backward pass is grown
    runner.opts.batchSerializationSettings.transformContext =
        BatchSerializationTransformContext::Bwd;
    runner.opts.batchSerializationSettings.method =
        BatchSerializationMethod::Loop;
    // Disable outlining (tested separately)
    runner.opts.enableOutlining  = false;
    runner.opts.virtualGraphMode = VirtualGraphMode::Manual;
    runner.patterns              = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.enableInPlace(false);
    runner.loss = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization with
  // loops:
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

    size_t numLoops = 0;

    for (size_t i = 0; i < schedule.size(); i++) {
      Op *op = schedule.at(i);
      logging::trace("Op: {} {} {}",
                     op->hasBatchSerializedPhase()
                         ? std::to_string(op->getBatchSerializedPhase())
                         : "*",
                     op->settings.schedulePriority,
                     op->debugName());

      if (op->isConvertibleTo<LoopOp>()) {
        ++numLoops;
      }
    }

    BOOST_CHECK_EQUAL(numLoops, 11);
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

    runner.opts.batchSerializationSettings.factor = K;
    // Enable outlining with no restrictions
    runner.opts.explicitRecomputation          = false;
    runner.opts.enableOutlining                = true;
    runner.opts.outlineThreshold               = -1.0;
    runner.opts.enableOutliningCopyCostPruning = false;
    runner.opts.virtualGraphMode               = VirtualGraphMode::Manual;
    runner.patterns = Patterns(PatternsLevel::Default);
    // Disable so that no false negatives (rhs vs. lhs inplace) exist
    runner.patterns.enableInPlace(false);
    runner.loss = loss;

    return act;
  });

  // Testing that the schedule is as expected for batch serialization:
  runner.checkIr([&](Ir &ir) {
    std::vector<Op *> schedule =
        ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

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
      BOOST_CHECK_EQUAL(graphIdAndCount.second, 2);
    }
  });
}

BOOST_AUTO_TEST_CASE(NllBatchSerializedTest) {
  auto getFinalWeights = [](int batchesPerStep,
                            const std::array<float, 6 * 2>
                                &vWeight, // initial weights
                            ReductionType r0,
                            ReductionType r1,
                            bool batchSerialize) {
    std::vector<std::vector<float>> readBackData;
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    int batchSize = 4;

    // data
    TensorInfo dataBatchInfo{"FLOAT", std::vector<int64_t>{batchSize, 4, 6}};

    TensorInfo dataStepInfo{
        "FLOAT", std::vector<int64_t>{batchesPerStep, batchSize, 4, 6}};
    auto input = builder->addInputTensor(dataBatchInfo);

    // weights
    TensorInfo weightInfo{"FLOAT", std::vector<int64_t>{1, 6, 2}};
    ConstVoidData cvWeight = {vWeight.data(), weightInfo};
    auto weight            = builder->addInitializedInputTensor(cvWeight);

    // labels 0 and 1
    TensorInfo labelBatchInfo{"INT32", std::vector<int64_t>{batchSize}};
    TensorInfo labelStepInfo{"INT32",
                             std::vector<int64_t>{batchesPerStep, batchSize}};

    auto mmOut = aiOnnx.matmul({input, weight});

    auto slice0 =
        aiOnnx.slice({mmOut}, {batchSize, 4, 1}, {0, 0, 0}, {0, 1, 2}, "slc0");
    auto reshape0 =
        builder->reshape_const(aiOnnx, {slice0}, {batchSize, 4}, "rsh0");
    auto sm0    = aiOnnx.softmax({reshape0}, 1, "sm0");
    auto label0 = builder->addInputTensor(labelBatchInfo);
    auto nll0   = aiGraphcore.nllloss({sm0, label0}, r0);

    auto slice1 =
        aiOnnx.slice({mmOut}, {batchSize, 4, 2}, {0, 0, 1}, {0, 1, 2}, "slc1");
    auto reshape1 =
        builder->reshape_const(aiOnnx, {slice1}, {batchSize, 4}, "rsh1");
    auto sm1    = aiOnnx.softmax({reshape1}, 1, "sm1");
    auto label1 = builder->addInputTensor(labelBatchInfo);
    auto nll1   = aiGraphcore.nllloss({sm1, label1}, r1);

    if (r0 == ReductionType::NoReduction) {
      nll0 = aiOnnx.reducesum({nll0}, std::vector<int64_t>{0}, false);
    }

    if (r1 == ReductionType::NoReduction) {
      nll1 = aiOnnx.reducesum({nll1}, std::vector<int64_t>{0}, false);
    }

    auto finalLoss = aiOnnx.sum({nll0, nll1});

    auto device = createTestDevice(TEST_TARGET, 1, 20);

    auto sessionOptions = SessionOptions();

    if (batchSerialize) {
      sessionOptions.batchSerializationSettings.factor = batchSize;
    }

    sessionOptions.enableOutlining = false;

    auto session = popart::TrainingSession::createFromOnnxModel(
        builder->getModelProto(),
        DataFlow(batchesPerStep, {{finalLoss, AnchorReturnType("All")}}),
        finalLoss,
        ConstSGD(1.0),
        device,
        InputShapeInfo(),
        sessionOptions,
        popart::Patterns(PatternsLevel::Default)
            .enableNlllWithSoftMaxGradDirect(false)
            .enableSoftMaxGradDirect(false));

    auto sched =
        session->getIr().getOpSchedule({}, RequireOptimalSchedule::Yes);

    int num_nll  = 0;
    int num_sum  = 0;
    int num_mean = 0;

    for (Op *op : sched) {
      logging::trace(op->debugName());
      if (op->isConvertibleTo<NllOp>()) {
        ++num_nll;
      }
      if (op->isConvertibleTo<SumOp>()) {
        ++num_sum;
      }
      if (op->isConvertibleTo<MeanOp>()) {
        ++num_mean;
      }
    }

    if (batchSerialize) {
      BOOST_CHECK_EQUAL(num_nll, 2 * batchSize);

      int expected_num_sum  = 0;
      int expected_num_mean = 0;

      if (r0 == ReductionType::Sum || r0 == ReductionType::NoReduction) {
        ++expected_num_sum;
      } else if (r0 == ReductionType::Mean) {
        ++expected_num_mean;
      }

      if (r1 == ReductionType::Sum || r1 == ReductionType::NoReduction) {
        ++expected_num_sum;
      } else if (r1 == ReductionType::Mean) {
        ++expected_num_mean;
      }
      BOOST_CHECK_EQUAL(num_sum, expected_num_sum);
      BOOST_CHECK_EQUAL(num_mean, expected_num_mean);

    } else {
      BOOST_CHECK_EQUAL(num_nll, 2);
      BOOST_CHECK_EQUAL(num_sum, 0);
      BOOST_CHECK_EQUAL(num_mean, 0);
    }

    logging::trace("IR OPS: {} {} {}", num_nll, num_sum, num_mean);

    session->prepareDevice();

    auto seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(0.f, 0.5f);
    UniformIntDistribution<int> idis(0, 3);

    WeightsIO weightsRead;
    std::vector<float> readBackWeights(weightInfo.nelms(), -777.0f);
    weightsRead.insert(weight, {readBackWeights.data(), weightInfo});

    std::vector<float> vInput(dataStepInfo.nelms(), 0.1);
    popart::NDArrayWrapper<float> inputWrapper(vInput.data(), dataStepInfo);
    for (auto &v : vInput) {
      v = fdis(eng);
    }

    std::vector<int> vLabel0(labelStepInfo.nelms(), 2);
    popart::NDArrayWrapper<int> label0Wrapper(vLabel0.data(), labelStepInfo);
    for (auto &v : vLabel0) {
      v = idis(eng);
    }

    std::vector<int> vLabel1(labelStepInfo.nelms(), 3);
    popart::NDArrayWrapper<int> label1Wrapper(vLabel1.data(), labelStepInfo);
    for (auto &v : vLabel1) {
      v = idis(eng);
    }

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input, inputWrapper},
        {label0, label0Wrapper},
        {label1, label1Wrapper}};

    std::vector<float> finalLossVec(1, -777.0f);
    popart::NDArrayWrapper<float> finalLossVecWrap(finalLossVec.data(),
                                                   {batchesPerStep});

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {finalLoss, finalLossVecWrap}};

    popart::StepIO stepio(inputs, anchors);

    session->weightsFromHost();

    session->run(stepio);
    session->weightsToHost();
    session->readWeights(weightsRead);

    readBackData.push_back(readBackWeights);
    readBackData.push_back(finalLossVec);

    return readBackData;
  };

  // generate random input data
  auto seed = 1011;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(0.f, 0.5f);
  std::array<float, 6 * 2> vWeight;
  for (auto &val : vWeight) {
    val = fdis(eng);
  }

  auto check = [](std::vector<std::vector<float>> weights0,
                  std::vector<std::vector<float>> weights1) {
    for (size_t idx = 0; idx < weights0.size(); ++idx) {
      float absErr = 0.0f;
      for (int i = 0; i < weights0[idx].size(); ++i) {
        absErr += std::abs(weights0[idx][i] - weights1[idx][i]);
        BOOST_CHECK(weights0[idx][i] != -777.0f);
        BOOST_CHECK(weights1[idx][i] != -777.0f);
      }
      std::cout << "Absolute error is " << absErr << std::endl;
      BOOST_CHECK(absErr < 1e-6);
    }
  };

  auto weightsn   = getFinalWeights(2,
                                  vWeight,
                                  ReductionType::NoReduction,
                                  ReductionType::NoReduction,
                                  false);
  auto weightsnbs = getFinalWeights(
      2, vWeight, ReductionType::NoReduction, ReductionType::NoReduction, true);
  check(weightsn, weightsnbs);

  auto weightss = getFinalWeights(
      2, vWeight, ReductionType::NoReduction, ReductionType::Sum, false);
  auto weightssbs = getFinalWeights(
      2, vWeight, ReductionType::NoReduction, ReductionType::Sum, true);
  check(weightss, weightssbs);

  auto weightsm = getFinalWeights(
      2, vWeight, ReductionType::NoReduction, ReductionType::Mean, false);
  auto weightsmbs = getFinalWeights(
      2, vWeight, ReductionType::NoReduction, ReductionType::Mean, true);
  check(weightsm, weightsmbs);
}

BOOST_AUTO_TEST_CASE(TestBatchSerialWithOverlappedSchedule) {
  // Run with remote tensors and with an overlapped batch serialisation schedule
  // and check the IR ordering is (some may appear multiple times):
  //
  //     RemoteLoad batch n
  //     IoTileCopy n
  //     Compute batch n
  //     RemoteLoad batch n+1
  //     IoTileCopy (I/O -> compute batch n+1)
  //     IoTileCopy (compute -> I/O batch n)
  //     RemoteStore batch n
  //     Compute batch n+1
  //     RemoteLoad batch n+2
  //     IoTileCopy (I/O -> compute batch n+2)
  //     IoTileCopy (compute -> I/O batch n+1)
  //     RemoteStore batch n+1
  //     Compute batch n+2
  //     etc.

  auto isRemoteLoad = [](Op *op, int bsp) {
    return (op->isConvertibleTo<RemoteLoadOp>()) &&
           (op->getBatchSerializedPhase() == bsp);
  };
  auto isIoTileCopyToCompute = [](Op *op, int bsp) {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::Compute) &&
           (op->getBatchSerializedPhase() == bsp);
  };
  auto isCompute = [](Op *op, int bsp) {
    return (!op->isConvertibleTo<RemoteLoadOp>()) &&
           (!op->isConvertibleTo<IoTileCopyOp>()) &&
           (!op->isConvertibleTo<RemoteStoreOp>()) &&
           (op->getBatchSerializedPhase() == bsp);
  };
  auto isIoTileCopyToIo = [](Op *op, int bsp) {
    return (op->isConvertibleTo<IoTileCopyOp>()) &&
           (op->settings.tileSet == TileSet::IO) &&
           (op->getBatchSerializedPhase() == bsp);
  };
  auto isRemoteStore = [](Op *op, int bsp) {
    return (op->isConvertibleTo<RemoteStoreOp>()) &&
           (op->getBatchSerializedPhase() == bsp);
  };
  auto isMultiExchange = [](Op *op, int bsp) {
    return (op->isConvertibleTo<MultiExchangeOp>()) &&
           (op->getBatchSerializedPhase() == bsp);
  };

  auto opFilter = [](Op *op, int phase) {
    return (op->hasBatchSerializedPhase()) &&
           (op->getBatchSerializedPhase() >= 0) && (op->hasExecutionPhase()) &&
           (op->getExecutionPhase() == phase);
  };

  auto test = [&](BatchSerializationBatchSchedule batchSchedule) {
    TestRunner runner;
    runner.isTraining = true;
    int N             = 4;   // layers.
    int M             = 8;   // data size
    int K             = 4;   // batch serialisation factor
    int size          = 100; // data size

    runner.buildModel([&](auto &builder) {
      auto aiOnnx = builder.aiOnnxOpset9();
      TensorInfo inInfo{"FLOAT", std::vector<int64_t>{M, size}};
      auto act = builder.addInputTensor(inInfo, "input");

      int phase = 0;

      for (int n = 0; n < N; ++n) {
        TensorInfo wInfo{"FLOAT", std::vector<int64_t>{size, size}};
        std::vector<TestTensor> inputs;
        std::vector<TestTensor> outputs;
        std::vector<float> wData(wInfo.nelms(), 0);
        ConstVoidData wCVData{wData.data(), wInfo};
        auto w = builder.addInitializedInputTensor(
            wCVData, logging::format("WEIGHTS[layer={} {}]", n, phase));
        act = aiOnnx.matmul({act, w},
                            logging::format("MATMUT[layer={}-{}]", n, phase));
        builder.executionPhase(act, phase);
        act =
            aiOnnx.relu({act}, logging::format("RELU[layer={}-{}]", n, phase));
        builder.executionPhase(act, phase);
        phase += 4;
      }

      auto loss = builder.aiGraphcoreOpset1().l1loss({act}, 0.1);
      builder.executionPhase(loss, phase);

      // Make sure IO tiles are usd.
      runner.opts.numIOTiles = 192;

      runner.opts.batchSerializationSettings.factor        = K;
      runner.opts.batchSerializationSettings.batchSchedule = batchSchedule;
      runner.opts.batchSerializationSettings.concatOnVirtualGraphChange = false;
      runner.opts.batchSerializationSettings.concatOnExecutionPhaseChange =
          false;
      runner.opts.virtualGraphMode = VirtualGraphMode::ExecutionPhases;
      runner.opts.executionPhaseSettings.phases = (N + 1) * 4;
      runner.opts.executionPhaseSettings.stages = 1;
      runner.opts.executionPhaseSettings.activationIOSchedule =
          ExecutionPhaseIOSchedule::OnDemand;

      // Make sure activations are stored in streaming memory, loaded via IO
      // tiles.
      runner.opts.activationTensorLocationSettings.location.storage =
          TensorStorage::OffChip;
      runner.opts.activationTensorLocationSettings.location.loadTileSet =
          TileSet::IO;
      runner.opts.activationTensorLocationSettings.location.storageTileSet =
          TileSet::IO;
      runner.opts.activationTensorLocationSettings.location
          .replicatedTensorSharding = ReplicatedTensorSharding::Off;
      runner.opts.activationTensorLocationSettings.minElementsForOffChip = 0;

      // Enable outlining with no restrictions
      runner.opts.explicitRecomputation          = false;
      runner.opts.enableOutlining                = false;
      runner.opts.outlineThreshold               = -1.0;
      runner.opts.enableOutliningCopyCostPruning = false;
      runner.patterns = Patterns(PatternsLevel::Default);

      // Disable so that no false negatives (rhs vs. lhs inplace) exist
      runner.patterns.enableInPlace(false);
      runner.loss = loss;

      return act;
    });

    // Testing that the schedule is as expected for batch serialization:
    runner.checkIr([&](Ir &ir) {
      std::vector<Op *> schedule =
          ir.getMainGraph().getOpSchedule({}, RequireOptimalSchedule::Yes);

      // Let's grab all some forward phase and look at the ops order in detail.
      std::vector<Op *> someFwdPhase;
      std::copy_if(schedule.begin(),
                   schedule.end(),
                   std::back_inserter(someFwdPhase),
                   std::bind(opFilter, std::placeholders::_1, 4));

      // Print ops for debugging.
      // for (auto op : someFwdPhase) {
      //  logging::trace("Op: {} {} {} {}",
      //                 op->hasExecutionPhase()
      //                     ? std::to_string(op->getExecutionPhase())
      //                     : "*",
      //                 op->hasBatchSerializedPhase()
      //                     ? std::to_string(op->getBatchSerializedPhase())
      //                     : "*",
      //                 op->settings.schedulePriority,
      //                 op->debugName());
      //}

      auto it = someFwdPhase.begin();

      if (batchSchedule == BatchSerializationBatchSchedule::OverlapOnIo) {
        BOOST_CHECK(someFwdPhase.size() >= 36);

        // Check interleaved/overlapped order.
        BOOST_CHECK(isRemoteLoad(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isRemoteLoad(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isMultiExchange(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isRemoteLoad(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isMultiExchange(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isRemoteLoad(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isMultiExchange(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isMultiExchange(*it++, 3));

      } else if (batchSchedule ==
                 BatchSerializationBatchSchedule::OverlapOnCompute) {
        BOOST_CHECK(someFwdPhase.size() >= 34);

        // Check interleaved/overlapped order.
        BOOST_CHECK(isRemoteLoad(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isRemoteLoad(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isMultiExchange(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isMultiExchange(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isMultiExchange(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isMultiExchange(*it++, 3));
      }

      // Now let's grab all some backwards phase and look at the ops order in
      // detail.

      std::vector<Op *> someBwdPhase;
      std::copy_if(schedule.begin(),
                   schedule.end(),
                   std::back_inserter(someBwdPhase),
                   std::bind(opFilter, std::placeholders::_1, 34));

      // Print ops for debugging.
      // for (auto op : someBwdPhase) {
      //  logging::trace("Op: {} {} {} {}",
      //                 op->hasExecutionPhase()
      //                     ? std::to_string(op->getExecutionPhase())
      //                     : "*",
      //                 op->hasBatchSerializedPhase()
      //                     ? std::to_string(op->getBatchSerializedPhase())
      //                     : "*",
      //                 op->settings.schedulePriority,
      //                 op->debugName());
      //}

      it = someBwdPhase.begin();

      if (batchSchedule == BatchSerializationBatchSchedule::OverlapOnIo) {
        BOOST_CHECK(someBwdPhase.size() >= 72);

        BOOST_CHECK(isMultiExchange(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isMultiExchange(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isRemoteStore(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isMultiExchange(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isRemoteStore(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isMultiExchange(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isRemoteStore(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isRemoteStore(*it++, 3));

      } else if (batchSchedule ==
                 BatchSerializationBatchSchedule::OverlapOnCompute) {
        BOOST_CHECK(someBwdPhase.size() >= 70);

        BOOST_CHECK(isMultiExchange(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 0));
        BOOST_CHECK(isMultiExchange(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isCompute(*it++, 0));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 0));
        BOOST_CHECK(isMultiExchange(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isCompute(*it++, 1));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 1));
        BOOST_CHECK(isMultiExchange(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 2));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 2));
        BOOST_CHECK(isRemoteStore(*it++, 2));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isCompute(*it++, 3));
        BOOST_CHECK(isIoTileCopyToIo(*it++, 3));
        BOOST_CHECK(isRemoteStore(*it++, 3));
      }
    });
  };

  test(BatchSerializationBatchSchedule::OverlapOnIo);
  test(BatchSerializationBatchSchedule::OverlapOnCompute);
}
