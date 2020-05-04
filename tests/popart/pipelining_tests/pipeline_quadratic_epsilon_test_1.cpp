// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineQuadraticEpsilonTest1

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <map>
#include <random>
#include <tuple>
#include <vector>

#define protected public
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/restore.hpp>
#include <popart/op/stash.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>
#undef protected

// A test of quadratic convergence for continuous pipelining.
// Comparison between
// {exact, continuous without recompute, continuous with recompute}
// for a model which is a chain of 6 of these:
//
//  -+
//   |
//   +-In------+    W
//     |       |    |
//    sin      |    |
//      \      |    |
//       \     |    |
//         add      |
//           \      |
//            \     |
//             \    |
//               add
//                |
//                |
//               Out
//                |
//                +-+
//                  |
//
// 2 layers on each of 3 IPUs.

BOOST_AUTO_TEST_CASE(QuadraticEpsilonTest1) {

  // dump poplar execution traces to file
  bool dumpPoplarTrace = false;

  // Store the initial and final weights from training
  class WeightPair {
  public:
    using Wts = std::vector<std::vector<float>>;
    WeightPair(Wts s, Wts e) : start(s), end(e) {}
    Wts start;
    Wts end;
  };

  enum class RunType {
    SingleDevice = 0,    // exact SGD on a single device
    ContinuousPipe,      // Continuous (in-exact) pipelining
    ContinuousRecompPipe // Continuous pipelined
  };

  using namespace popart;

  // Return a vector of {initial weights, updated weights},
  // one for each {learnRate, nSteps}.
  // Note: This call only compiles one engine, so adding new learning
  // rates does not increase compile time too considerably
  auto getResults = [dumpPoplarTrace](RunType rt,
                                      const std::vector<float> &learnRates,
                                      const std::vector<int64_t> &nSteps_s,
                                      int accumulationFactor) {
    if (learnRates.size() != nSteps_s.size()) {
      throw error("learnRates and nSteps_s not of same size, invalid test");
    }

    int microBatchesPerStep = 20;
    if (microBatchesPerStep % accumulationFactor != 0) {
      throw error(
          "accumulation factor {} is not a factor of microBatchesPerStep {}",
          accumulationFactor,
          microBatchesPerStep);
    }

    // The number of weight updates per step
    int64_t batchesPerStep = microBatchesPerStep / accumulationFactor;

    int seed = 1011;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(-1, 1);
    int64_t microBatchSize = 3;
    int64_t sampleHeight   = 8;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};
    std::vector<int64_t> weightShape = sampleShape;

    // The input "image" processed by the network has this shape
    std::vector<int64_t> microBatchShape{
        microBatchSize, sampleHeight, sampleHeight};

    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo microBatchInfo{"FLOAT", microBatchShape};
    int64_t sampleElms{sampleHeight * sampleHeight};

    // number of "pixels" in a micro-batch
    int64_t microBatchElms = sampleElms * microBatchSize;

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(microBatchInfo, "0tupni");

    // Storage of necessary weights, ids, and pointers; for all layers.
    std::vector<std::vector<float>> allWeights;
    std::vector<ConstVoidData> allWeightCvds;
    std::vector<TensorId> allWeightIds;
    std::vector<TensorId> actIds;
    std::vector<std::vector<float>> w_readbacks;
    WeightsIO weightsRead;

    // For the 2 pipeline cases, use multiple IPUs.
    int nIPUs = (rt != RunType::SingleDevice ? 3 : 1);

    auto addLayer = [&allWeights,
                     &allWeightCvds,
                     &allWeightIds,
                     &actIds,
                     sampleElms,
                     &weightsRead,
                     weightInfo,
                     sampleInfo,
                     &fdis,
                     &eng,
                     &builder,
                     &aiOnnx,
                     &w_readbacks](TensorId actInId, int vgid) {
      // The weights which will be readback for this layer are stored here
      w_readbacks.push_back(std::vector<float>(sampleElms, -99.0f));

      // The initializer weights for this layer are stored here
      allWeights.push_back(std::vector<float>(sampleElms, 0));

      // initialize weights for this layer
      for (auto &x : allWeights.back()) {
        x = 100 * fdis(eng);
      }

      // store pointers and IDs
      allWeightCvds.push_back({allWeights.back().data(), sampleInfo});
      allWeightIds.push_back(
          builder->addInitializedInputTensor(allWeightCvds.back()));

      //
      // add weights and apply small non-linear bifurcation:
      // x -> (x + sin(x) + w)
      //

      auto act0 = aiOnnx.sin({actInId});
      builder->virtualGraph(act0, vgid);

      act0 = aiOnnx.add({act0, actInId});
      builder->virtualGraph(act0, vgid);
      // To prevent the cos Ops (grad of sin) from running in the fwd pass, we
      // must prevent the adds from being inplace. This ensures the stashing we
      // need to test
      builder->setInplacePreferences(
          act0, {{"AddLhsInplace", -100.0f}, {"AddRhsInplace", -100.0f}});

      act0 = aiOnnx.add({act0, allWeightIds.back()});
      builder->virtualGraph(act0, vgid);
      builder->setInplacePreferences(
          act0, {{"AddLhsInplace", -100.0f}, {"AddRhsInplace", -100.0f}});

      actIds.push_back(act0);
      weightsRead.insert(allWeightIds.back(),
                         {w_readbacks.back().data(), weightInfo});
    };

    int nLayers = 6;
    addLayer(input0, 0);
    for (int i = 1; i < nLayers; ++i) {
      addLayer(actIds.back(), (i * nIPUs) / nLayers);
    }
    TensorId actFinal = actIds.back();

    SessionOptions userOptions;
    std::map<std::string, std::string> deviceOpts{
        {"numIPUs", std::to_string(nIPUs)}};

    userOptions.enableVirtualGraphs = true;

    if (dumpPoplarTrace) {
      userOptions.reportOptions.insert({"showExecutionSteps", "true"});
    }

    if (rt != RunType::SingleDevice) {
      userOptions.enablePipelining = true;
    }

    if (rt == RunType::ContinuousRecompPipe) {
      userOptions.autoRecomputation = RecomputationType::Standard;
    }

    if (accumulationFactor > 1) {
      userOptions.accumulationFactor         = accumulationFactor;
      userOptions.enableGradientAccumulation = true;
    } else {
      userOptions.enableGradientAccumulation = false;
    }

    builder->addOutputTensor(actFinal);
    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(batchesPerStep);

    // The learning rate will be adjusted to the correct value at runtime
    auto optimizer = SGD({{"defaultLearningRate", {10000., false}}});

    float lambda = 0.1;
    auto loss    = std::unique_ptr<Loss>(
        new L1Loss(actFinal, "l1LossVal", lambda, ReductionType::Sum));
    loss->virtualGraph(nIPUs - 1);

    // number of "pixels" in a step
    int64_t stepDataElms = accumulationFactor * microBatchElms * batchesPerStep;

    auto device = createTestDevice(TEST_TARGET, nIPUs);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        {loss.get()},
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    auto opSchedule = session->ir.getOpSchedule({});
    int nRecomp     = 0;
    for (auto op : opSchedule) {
      if (op->settings.recomputeType == RecomputeType::Recompute) {
        ++nRecomp;
      }
    }

    // Note that having Ops annotated as recompute does not, for pipelining,
    // imply anthing will be recomputed. The true test for this will be that the
    // number of Tensors stashed is reduced for the recompute in pipelining
    // case.
    if (rt == RunType::ContinuousRecompPipe) {
      BOOST_CHECK(nRecomp > 0);
    } else {
      BOOST_CHECK(nRecomp == 0);
    }

    session->prepareDevice();

    // all the data for a single step
    std::vector<float> v_input_x(stepDataElms);

    std::map<popart::TensorId, popart::IArray &> anchors = {};

    std::vector<WeightPair> weightPairs;
    for (int iteration = 0; iteration < learnRates.size(); ++iteration) {

      // Fix the seed, to ensure the same sequence of sample generation for each
      // learning rate
      eng.seed(57);

      auto learnRate = learnRates.at(iteration);
      auto nSteps    = nSteps_s.at(iteration);

      // initialize weights, write weights to device
      session->resetHostWeights(proto);
      session->weightsFromHost();

      // initialize learnRate, write to device
      SGD newOptimizer({{"defaultLearningRate", {learnRate, false}}});
      session->updateOptimizer(&newOptimizer);
      session->optimizerFromHost();

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

        if (dumpPoplarTrace) {
          if (iteration == 0 && i == 0 && rt == RunType::ContinuousRecompPipe) {
            auto summaryReport = session->getSummaryReport();
            std::ofstream filly;
            filly.open("recompLog.log");
            filly << summaryReport;
            filly.close();
          }

          if (iteration == 0 && i == 0 && rt == RunType::ContinuousPipe) {
            auto summaryReport = session->getSummaryReport();
            std::ofstream filly;
            filly.open("pipelineLog.log");
            filly << summaryReport;
            filly.close();
          }
        }
      }

      // read final weights back
      session->weightsToHost();
      session->readWeights(weightsRead);
      weightPairs.push_back(WeightPair(allWeights, w_readbacks));
    }

    return weightPairs;
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

    std::cout << "| exact end - exact start |_1    "
              << "   | exact_end - continuous_end |_1 / "
              << " | exact_end - exact_start |_1 " << std::endl;

    for (int i = 0; i < delta_exact.size(); ++i) {
      std::cout << delta_exact[i] << "   " << delta_ends[i] / delta_exact[i]
                << std::endl;
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

  std::vector<float> learnRates{
      0.128, 0.064, 0.032, 0.016, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.128};

  std::vector<int64_t> nSteps_s{1, 2, 4, 8, 16, 32, 64, 128, 256, 1};

  int accumulationFactor = 1;

  std::cout << "\nGet results for Continuous without Recomputation\n"
            << std::endl;
  auto continuous = getResults(
      RunType::ContinuousPipe, learnRates, nSteps_s, accumulationFactor);

  std::cout << "\nGet results for Continuous with Recomputation\n" << std::endl;
  auto recomp = getResults(
      RunType::ContinuousRecompPipe, learnRates, nSteps_s, accumulationFactor);

  std::cout << "\nGet results for SingleDevice run\n" << std::endl;
  auto exact = getResults(
      RunType::SingleDevice, learnRates, nSteps_s, accumulationFactor);

  bool printLog = true;

  if (printLog) {
    for (int i = 0; i < exact.size() - 1; ++i) {
      std::cout << "\n@ learning rate " << learnRates[i] << std::endl;
      std::cout << "-----------------" << std::endl;
      std::cout << "For continuous (without recompute)" << std::endl;
      print(exact[i], continuous[i]);
      std::cout << "For continuous (with recompute)" << std::endl;
      print(exact[i], recomp[i]);
    }

    std::cout << "Reporting mean relative differences over all weights "
              << std::endl;
    for (int i = 0; i < exact.size() - 1; ++i) {
      std::cout << "\n@ learning rate " << learnRates[i] << std::endl;
      std::cout << "exact <-> continuous "
                << getMeanRelative(exact[i], continuous[i]) << std::endl;

      std::cout << "exact <-> recomp " << getMeanRelative(exact[i], recomp[i])
                << std::endl;

      std::cout << "continuous <-> recomp "
                << getMeanRelative(continuous[i], recomp[i]) << std::endl;
    }
  }

  // The final run should match the first run exactly (same learning rate)
  BOOST_CHECK(getMeanRelative(exact.back(), exact[0]) < 1e-8);

  // at 0.0005:
  auto small = getMeanRelative(exact[8], continuous[8]);

  // at 0.128:
  auto large = getMeanRelative(exact[0], continuous[0]);

  //  the experiment above give 0.11868 for this run
  BOOST_CHECK(large < 0.5);

  // the ratio should be about 2**8 using, using a margin of 2**2
  BOOST_CHECK(large / small > (2 << 6));
}

// very nice convergence (04 September 2019):

// 165: @ learning rate 0.128
// 165: exact <-> continuous 0.143023
// 165: exact <-> recomp 0.137421
// 165: continuous <-> recomp 0.0168884
// 165:
// 165: @ learning rate 0.064
// 165: exact <-> continuous 0.0598955
// 165: exact <-> recomp 0.0593508
// 165: continuous <-> recomp 0.00738907
// 165:
// 165: @ learning rate 0.032
// 165: exact <-> continuous 0.0276319
// 165: exact <-> recomp 0.0270495
// 165: continuous <-> recomp 0.00313846
// 165:
// 165: @ learning rate 0.016
// 165: exact <-> continuous 0.0132138
// 165: exact <-> recomp 0.012881
// 165: continuous <-> recomp 0.00172954
// 165:
// 165: @ learning rate 0.008
// 165: exact <-> continuous 0.00570888
// 165: exact <-> recomp 0.00564257
// 165: continuous <-> recomp 0.000833319
// 165:
// 165: @ learning rate 0.004
// 165: exact <-> continuous 0.00257001
// 165: exact <-> recomp 0.00253338
// 165: continuous <-> recomp 0.000424094
// 165:
// 165: @ learning rate 0.002
// 165: exact <-> continuous 0.00119873
// 165: exact <-> recomp 0.00115962
// 165: continuous <-> recomp 0.000205215
// 165:
// 165: @ learning rate 0.001
// 165: exact <-> continuous 0.000569798
// 165: exact <-> recomp 0.000562985
// 165: continuous <-> recomp 0.000106053
// 165:
// 165: @ learning rate 0.0005
// 165: exact <-> continuous 0.00030137
// 165: exact <-> recomp 0.000299084
// 165: continuous <-> recomp 5.77406e-05
//
