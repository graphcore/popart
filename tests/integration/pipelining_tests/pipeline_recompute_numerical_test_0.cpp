// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineRecomputeNumericalTest0

#include <../random_util.hpp>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

#define protected public
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>
#undef protected
#include <popart/session.hpp>

BOOST_AUTO_TEST_CASE(PipelineRecomputeNumericalTest0x) {

  // Model :
  //
  //  Input         Weights
  //     |            |
  //     +--- mul ----+
  //           |
  //         sigmoid
  //           |
  //          scale
  //           |            IPU 0
  // - -  - -  - -  - -  - -  - -  - -  - -  - -  - -
  //           |            IPU 1
  //          L1 loss
  //
  //
  //  Without recomputation, both Input and output of sigmoid are stashed.
  //  Note that Sigmoid grad uses the output of sigmoid - this is whyoutput if
  //  sigmoid must be kept.
  //
  //  With recomputation, the output of sigmoid is not stashed, but recomputed
  //  in the backwards pass.
  //
  //
  //
  //
  //  Test :
  //
  //  Using gradient accumulation (3 microbatches accumulated, 1 batch per step)
  //  We confirm that
  //  1) no pipelining
  //  2) pipelining without recomputation
  //  3) pipelining with recomputation
  //  all produce exactly the same final model (after training for 2 steps)
  //
  //  We also confirm that the weights have actually changed, and that they are
  //  not fully converged

  // Store the initial and final weights from training
  class WeightPair {
  public:
    using Wts = std::vector<std::vector<float>>;
    WeightPair(const Wts &s, const Wts &e) : start(s), end(e) {}
    Wts start;
    Wts end;
  };

  enum class RunType {
    SingleDevice = 0,    // exact SGD on a single device
    ContinuousPipe,      // Continuous (in-exact) pipelining
    ContinuousRecompPipe // Continuous pipelined
  };

  using namespace popart;

  auto getResult = [](RunType rt) {
    int accumulationFactor  = 3;
    int microBatchesPerStep = 3;
    if (microBatchesPerStep % accumulationFactor != 0) {
      throw error("accumulation factor is not a factor of microBatchesPerStep");
    }
    int64_t batchesPerStep = microBatchesPerStep / accumulationFactor;

    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-1.f, +1.f);

    int64_t microBatchSize = 2;
    int64_t sampleHeight   = 8;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};
    std::vector<int64_t> weightShape = sampleShape;
    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t microBatchElms = sampleElms * microBatchSize;

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(microBatchInfo, "0tupni");

    WeightsIO weights_io_read;
    int nIPUs = (rt != RunType::SingleDevice ? 2 : 1);

    auto weights_readback = std::vector<float>(sampleElms, -99.0f);
    auto weights_init     = std::vector<float>(sampleElms, 0);
    for (auto &x : weights_init) {
      x = 1 * fdis(eng);
    }
    ConstVoidData weights_cvd = {weights_init.data(), sampleInfo};
    TensorId weights_id       = builder->addInitializedInputTensor(weights_cvd);

    auto act0 = aiOnnx.mul({weights_id, input0});
    builder->virtualGraph(act0, 0);

    act0 = aiOnnx.sigmoid({act0});
    builder->virtualGraph(act0, 0);

    act0 = aiGraphcore.scale({act0}, 2.0f);
    builder->virtualGraph(act0, 0);

    weights_io_read.insert(weights_id, {weights_readback.data(), weightInfo});

    TensorId actFinal = act0;

    SessionOptions userOptions;

    userOptions.virtualGraphMode = VirtualGraphMode::Manual;

    userOptions.reportOptions.insert({"showExecutionSteps", "true"});

    if (rt != RunType::SingleDevice) {
      userOptions.enablePipelining = true;
    }

    if (rt == RunType::ContinuousRecompPipe) {
      userOptions.autoRecomputation = RecomputationType::Standard;
    }

    if (accumulationFactor > 1) {
      userOptions.accumulationFactor         = accumulationFactor;
      userOptions.enableGradientAccumulation = true;
    }

    // Changing this to SGD does not work, learning rate is not correct I think.
    auto optimizer = ConstSGD(0.04);

    float lambda = 0.159;
    actFinal     = builder->aiGraphcoreOpset1().l1loss({actFinal}, lambda);
    builder->virtualGraph(actFinal, nIPUs - 1);

    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(batchesPerStep);

    int64_t stepDataElms = accumulationFactor * microBatchElms * batchesPerStep;

    auto device = createTestDevice(TEST_TARGET, nIPUs);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        actFinal,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    auto opSchedule =
        session->getIr().getOpSchedule({}, RequireOptimalSchedule::Yes);
    int nRestore = 0;
    for (auto op : opSchedule) {
      // number of restores
      if (dynamic_cast<RestoreOp *>(op) ||
          dynamic_cast<RestoreInplaceOp *>(op)) {
        ++nRestore;
      }
    }
    if (rt == RunType::SingleDevice) {
      BOOST_CHECK(nRestore == 0);
    } else if (rt == RunType::ContinuousRecompPipe) {
      BOOST_CHECK(nRestore == 1);
    } else {
      BOOST_CHECK(rt == RunType::ContinuousPipe);
      BOOST_CHECK(nRestore == 2);
    }

    session->prepareDevice();

    std::vector<float> v_input_x(stepDataElms);
    std::map<popart::TensorId, popart::IArray &> anchors = {};

    std::vector<WeightPair> weightPairs;

    eng.seed(57);

    auto nSteps = 2;

    // initialize weights, write weights to device
    session->resetHostWeights(proto);
    session->weightsFromHost();

    int64_t samplesPerStep =
        batchesPerStep * accumulationFactor * microBatchSize;

    std::vector<int64_t> stepDataShape{batchesPerStep * accumulationFactor,
                                       microBatchSize,
                                       sampleHeight,
                                       sampleHeight};

    TensorInfo stepDataInfo{"FLOAT", stepDataShape};

    for (int i = 0; i < nSteps; ++i) {
      std::cout << "Step # " << i << std::endl;

      // generate new samples
      for (int i = 0; i < samplesPerStep; ++i) {
        for (int j = 0; j < sampleElms; ++j) {
          auto stepIndex       = i * sampleElms + j;
          v_input_x[stepIndex] = fdis(eng);
        }
      }
      popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                    stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input0, input_x_wrapper}};
      popart::StepIO stepio(inputs, anchors);

      // process the samples
      session->run(stepio);

      if (i == 0 && rt == RunType::ContinuousRecompPipe) {
        auto summaryReport = session->getSummaryReport();
        std::ofstream filly;
        filly.open("recompLog.log");
        filly << summaryReport;
        filly.close();
      }

      if (i == 0 && rt == RunType::ContinuousPipe) {
        auto summaryReport = session->getSummaryReport();
        std::ofstream filly;
        filly.open("pipelineLog.log");
        filly << summaryReport;
        filly.close();
      }
    }

    // read final weights back
    session->weightsToHost();
    session->readWeights(weights_io_read);
    return WeightPair({weights_init}, {weights_readback});
  };

  auto getDelta = [](const std::vector<std::vector<float>> &a,
                     const std::vector<std::vector<float>> &b) {
    std::vector<float> vDelta;
    for (int layer = 0; layer < a.size(); ++layer) {
      float delta = 0;
      for (int i = 0; i < a[layer].size(); ++i) {
        delta += std::abs(a[layer][i] - b[layer][i]);
      }
      vDelta.push_back(delta);
    }
    return vDelta;
  };

  auto print = [&getDelta](const WeightPair &exact,
                           const WeightPair &continuous) {
    auto delta_exact      = getDelta(exact.start, exact.end);
    auto delta_continuous = getDelta(continuous.start, continuous.end);
    auto delta_starts     = getDelta(continuous.start, exact.start);
    auto delta_ends       = getDelta(continuous.end, exact.end);

    std::cout << "(i) | exact end - exact start |_1    (ii) | exact_end - "
                 "continuous_end |_1 / | exact_end - exact_start |_1 "
              << std::endl;
    for (int i = 0; i < delta_exact.size(); ++i) {
      std::cout << "(i) " << delta_exact[i] << "     (ii) "
                << delta_ends[i] / delta_exact[i] << std::endl;
    }
  };

  auto getMeanRelative = [&getDelta](const WeightPair &wp0,
                                     const WeightPair &wp1) {
    float sum       = 0.0f;
    auto delta_0    = getDelta(wp0.start, wp0.end);
    auto delta_ends = getDelta(wp0.end, wp1.end);
    for (auto i = 0; i < delta_0.size(); ++i) {
      sum += delta_ends[i] / delta_0[i];
    }
    return sum / delta_0.size();
  };

  std::cout << "Get results for Continuous without Recomputation\n\n"
            << std::endl;
  auto continuous = getResult(RunType::ContinuousPipe);
  std::cout << "Get results for Continuous with Recomputation\n\n" << std::endl;
  auto recomp = getResult(RunType::ContinuousRecompPipe);
  std::cout << "Get results for SingleDevice run\n\n" << std::endl;
  auto exact = getResult(RunType::SingleDevice);

  bool printLog = true;

  if (printLog) {
    std::cout << "For continuous (no recompute)" << std::endl;
    print(exact, continuous);

    std::cout << "The relative difference between exact and continuous is "
              << getMeanRelative(exact, continuous) << std::endl;

    std::cout << "The relative difference between exact and recomp is "
              << getMeanRelative(exact, recomp) << std::endl;
  }

  //  the experiment above give 0.11868 for this run
  BOOST_CHECK(getMeanRelative(exact, continuous) < 1e-8);
  BOOST_CHECK(getMeanRelative(exact, recomp) < 1e-8);
}
