#define BOOST_TEST_MODULE PipelineQuadraticEpsilonTest1

#include <algorithm>
#include <boost/test/unit_test.hpp>
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
#undef protected

// Another test of quadratic convergence for continuous pipelining. This test
// only compiles engines once, so adding new learning rates does not increase
// compile time too considerably

BOOST_AUTO_TEST_CASE(QuadraticEpsilonTest1) {

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
    ContinuousRecompPipe // Coming soon T9575
  };

  using namespace popart;

  // Return a vector of {initial weights, updated weights}, one for each
  // {learnRate, nSteps}.
  auto getResults = [](RunType rt,
                       std::vector<float> learnRates,
                       std::vector<int64_t> nSteps_s) {
    assert(learnRates.size() == nSteps_s.size());

    int seed = 1011;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(-1, 1);
    int64_t batchSize    = 4;
    int64_t sampleHeight = 8;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};
    std::vector<int64_t> weightShape = sampleShape;
    std::vector<int64_t> batchShape{batchSize, sampleHeight, sampleHeight};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo batchInfo{"FLOAT", batchShape};
    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t weightsElms = sampleElms;
    int64_t batchElms   = sampleElms * batchSize;
    int nIPUs           = (rt != RunType::SingleDevice ? 3 : 1);

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(batchInfo, "0tupni");

    // Storage for all layers
    std::vector<std::vector<float>> allWeights;
    std::vector<ConstVoidData> allWeightCvds;
    std::vector<TensorId> allWeightIds;
    std::vector<TensorId> actIds;
    std::vector<std::vector<float>> w_readbacks;
    WeightsIO weightsRead;

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
                     &aiGraphcore,
                     &w_readbacks](TensorId actInId, int vgid) {
      w_readbacks.push_back(std::vector<float>(sampleElms, -99.0f));
      allWeights.push_back(std::vector<float>(sampleElms, 0));

      for (auto &x : allWeights.back()) {
        x = 10 * fdis(eng);
      }

      allWeightCvds.push_back({allWeights.back().data(), sampleInfo});
      allWeightIds.push_back(
          builder->addInitializedInputTensor(allWeightCvds.back()));

      // add weights and apply small non-linear bifurcation
      // x -> x + sin(x) + w
      auto act0 = aiOnnx.sin({actInId});
      builder->virtualGraph(act0, vgid);
      act0 = aiOnnx.add({act0, actInId});
      builder->virtualGraph(act0, vgid);
      act0 = aiOnnx.add({act0, allWeightIds.back()});
      builder->virtualGraph(act0, vgid);
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

    // 100 weight updates per step, but the number of steps depends on the
    // learning-rate.
    int64_t batchesPerStep = 100;

    builder->addOutputTensor(actFinal);
    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(batchesPerStep, {});

    // The learning rate will be adjusted to the correct value at runtime
    auto optimizer = SGD(10000.);

    float lambda = 0.1;
    auto loss    = std::unique_ptr<Loss>(
        new L1Loss(actFinal, "l1LossVal", lambda, ReductionType::SUM));
    loss->virtualGraph(nIPUs - 1);

    SessionOptions userOptions;
    std::map<std::string, std::string> deviceOpts{
        {"numIPUs", std::to_string(nIPUs)}};

    userOptions.enableVirtualGraphs = true;

    if (rt != RunType::SingleDevice) {
      userOptions.enablePipelining = true;
    }

    if (rt == RunType::ContinuousRecompPipe) {
      userOptions.autoRecomputation = RecomputationType::Standard;
    }

    int64_t stepDataElms = batchElms * batchesPerStep;

    auto device =
        DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        {loss.get()},
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::DEFAULT));

    session->prepareDevice();

    std::vector<float> v_input_x(stepDataElms);
    std::map<popart::TensorId, popart::IArray &> anchors = {};

    std::vector<WeightPair> weightPairs;
    for (int iteration = 0; iteration < learnRates.size(); ++iteration) {

      // The same sequence of sample generation for each learning rate
      eng.seed(57);

      auto learnRate = learnRates.at(iteration);
      auto nSteps    = nSteps_s.at(iteration);

      // initialize weights, write weights to device
      session->resetHostWeights(proto);
      session->weightsFromHost();

      // initialize learnRate, write to device
      SGD newOptimizer(learnRate);
      session->updateOptimizer(&newOptimizer);
      session->optimizerFromHost();

      int64_t samplesPerStep = batchesPerStep * batchSize;

      std::vector<int64_t> stepDataShape{
          batchesPerStep, batchSize, sampleHeight, sampleHeight};
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
                           const WeightPair &continuous,
                           float lr) {
    auto delta_exact      = getDelta(exact.start, exact.end);
    auto delta_continuous = getDelta(continuous.start, continuous.end);
    auto delta_starts     = getDelta(continuous.start, exact.start);
    auto delta_ends       = getDelta(continuous.end, exact.end);

    std::cout << "@ learning rate = " << lr << std::endl;
    std::cout << "| exact end - exact start |_1     | exact_end - "
                 "continuous_end |_1 / | exact_end - exact_start |_1 "
              << std::endl;
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

  auto continuous = getResults(RunType::ContinuousPipe, learnRates, nSteps_s);
  auto exact      = getResults(RunType::SingleDevice, learnRates, nSteps_s);

  bool printLog = true;

  if (printLog) {

    for (int i = 0; i < exact.size(); ++i) {
      print(exact[i], continuous[i], learnRates[i]);
    }

    for (int i = 0; i < exact.size() - 1; ++i) {
      std::cout
          << "At learning rate " << learnRates[i]
          << " the difference between exact and continuous pipelining is  "
          << getMeanRelative(exact[i], continuous[i]) << std::endl;
    }

    // The final run should match the first run exactly (same learning rate)
    BOOST_CHECK(getMeanRelative(exact.back(), exact[0]) < 1e-8);

    // clang-format off
    // At learning rate 0.128 the difference between exact and continuous pipelining is  0.11868
    // At learning rate 0.064 the difference between exact and continuous pipelining is  0.0477695
    // At learning rate 0.032 the difference between exact and continuous pipelining is  0.0210121
    // At learning rate 0.016 the difference between exact and continuous pipelining is  0.0215663
    // At learning rate 0.008 the difference between exact and continuous pipelining is  0.00407066
    // At learning rate 0.004 the difference between exact and continuous pipelining is  0.00462797
    // At learning rate 0.002 the difference between exact and continuous pipelining is  0.00284805
    // At learning rate 0.001 the difference between exact and continuous pipelining is  0.000550351
    // At learning rate 0.0005 the difference between exact and continuous pipelining is  0.000214015
    // clang-format on

    // at 0.0005:
    auto small = getMeanRelative(exact[8], continuous[8]);

    // at 0.128:
    auto large = getMeanRelative(exact[0], continuous[0]);

    //  the experiment above give 0.11868 for this run
    BOOST_CHECK(large < 0.5);

    // the ratio should be about 2**8 using, using a margin of 2**2
    BOOST_CHECK(large / small > (2 << 6));
  }
}
