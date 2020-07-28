// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BatchSerializeIrTest

#include <../random_util.hpp>
#include <../test_runner.hpp>
#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/graph.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/sum.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#define protected public
#include <popart/session.hpp>
#undef protected

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
      sessionOptions.batchSerializationFactor = batchSize;
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

    auto sched = session->ir.getOpSchedule({});

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
      BOOST_ASSERT(num_nll == 2 * batchSize);

      int expected_num_sum  = 0;
      int expected_num_mean = 0;

      if (r0 == ReductionType::Sum) {
        ++expected_num_sum;
      } else if (r0 == ReductionType::Mean) {
        ++expected_num_mean;
      }

      if (r1 == ReductionType::Sum) {
        ++expected_num_sum;
      } else if (r1 == ReductionType::Mean) {
        ++expected_num_mean;
      }
      BOOST_ASSERT(num_sum == expected_num_sum);
      BOOST_ASSERT(num_mean == expected_num_mean);

    } else {
      BOOST_ASSERT(num_nll == 2);
      BOOST_ASSERT(num_sum == 0);
      BOOST_ASSERT(num_mean == 0);
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
      for (int i = 0; i < weights1.size(); ++i) {
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
