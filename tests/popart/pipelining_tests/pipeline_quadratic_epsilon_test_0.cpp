#define BOOST_TEST_MODULE PipelineQuadraticEpsilonTesto0

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

// The tricky thing with testing continuous pipelining is that we don't
// have a baseline, as it is not exact SGD. But "theory predicts" that as
// learning rate -> 0, continuous pipelining approaches exact SGD.
//
// Model, initialized with weights uniform from [-1, +1], inputs uniform from
// [0,4].
//
//   input1              input2
//     |                   |
//    (Add) <- Weight     (Add) <- Weight
//     |                   |
//  (Sigmoid)           (Sigmoid)
//     |                   |
//    (Add) <- Weight     (Add) <- Weight
//     |                   |
//  (Sigmoid)           (Sigmoid)
//     |                   |
//    (Add) <- Weight     (Add) <- Weight
//     |                   |
//  (Sigmoid)           (Sigmoid)
//     \                   |
//      \                  |
//       \----------(Add)--|
//                    |
//                    |
//                    |
//                 finalOut
//                    |
//                  l1-loss
//
//
// Results:
//
// l-rate  | n-batches | SGD_delta | cont_delta | SGD_cont_error |
// 0.04      20          2.15029     2.14938      0.00147016
// 0.02      40          2.1499      2.14942      0.000760384
// 0.01      80          2.15091     2.15066      0.000388138
// 0.005     160           .             .        0.000195466
//
// where
//
// SGD_delta  =  |weights_start - weights_end|_1 for exact SGD
// cont_delta =  |weights_start - weights_end|_1 for continuous pipelining
// SGD_cont_error = |weights_end_cont - weights_end_SGD|_1
//
// Notice the linear scaling of the error with learning rate
// (quadratic per step)
//
// THIS TEST:
// confirm that the final line is reproduced, that is, when
// l-rate  = 0.005, n-batches = 160, the error is less than 0.0003.

BOOST_AUTO_TEST_CASE(QuadraticEpsilolTest0) {

  // Store the initial and final weights from training
  class WeightPair {
  public:
    using Wts = std::vector<std::vector<float>>;
    WeightPair(Wts s, Wts e) : start(s), end(e) {}
    Wts start;
    Wts end;
  };

  using namespace popart;

  bool printStdOut = true;

  // Return {initial weights, updated weights}
  auto getResults = [](bool continuous) {
    // input stream samples weights are generated randomly
    int seed = 1011;
    std::default_random_engine eng(seed);
    std::uniform_real_distribution<float> fdis(-1, 1);

    int64_t batchSize      = 4;
    int64_t batchesPerStep = 160;
    int64_t sampleHeight   = 2;
    int64_t samplesPerStep = batchesPerStep * batchSize;
    std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};
    std::vector<int64_t> weightShape = sampleShape;
    std::vector<int64_t> batchShape{batchSize, sampleHeight, sampleHeight};
    std::vector<int64_t> stepDataShape{
        batchesPerStep, batchSize, sampleHeight, sampleHeight};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo batchInfo{"FLOAT", batchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};
    int64_t sampleElms{sampleHeight * sampleHeight};
    int64_t batchElms    = sampleElms * batchSize;
    int64_t stepDataElms = batchElms * batchesPerStep;

    // The model: see above

    // number of Adds on each of the two branches.
    int nLayers = 3;

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input0      = builder->addInputTensor(batchInfo, "0tupni");
    auto input1      = builder->addInputTensor(batchInfo, "1tupni");

    // Storage for all layers
    std::vector<std::vector<float>> allWeights;
    std::vector<ConstVoidData> allWeightCvds;
    std::vector<TensorId> allWeightIds;
    std::vector<TensorId> allSigActIds;
    std::vector<std::vector<float>> w_readbacks;
    WeightsIO weightsRead;

    int nLayersAdded    = 0;
    auto insertAddLayer = [&allWeights,
                           &allWeightCvds,
                           &allWeightIds,
                           &allSigActIds,
                           sampleElms,
                           &weightsRead,
                           weightInfo,
                           sampleInfo,
                           &fdis,
                           &eng,
                           &builder,
                           &aiOnnx,
                           &nLayersAdded,
                           &w_readbacks](TensorId actInId) {
      w_readbacks.push_back(std::vector<float>(sampleElms, -99.0f));
      allWeights.push_back(std::vector<float>(sampleElms, 0));
      for (auto &x : allWeights.back()) {
        x = fdis(eng);
      }
      allWeightCvds.push_back({allWeights.back().data(), sampleInfo});
      allWeightIds.push_back(
          builder->addInitializedInputTensor(allWeightCvds.back()));
      TensorId addActOutId = "addAct" + std::to_string(nLayersAdded);
      TensorId sigActOutId = "sigAct" + std::to_string(nLayersAdded);
      auto addOut = aiOnnx.add({allWeightIds.back(), actInId}, addActOutId);
      auto sigOut = aiOnnx.sigmoid({addOut}, sigActOutId);
      allSigActIds.push_back(sigOut);

      weightsRead.insert(allWeightIds.back(),
                         {w_readbacks.back().data(), weightInfo});
      ++nLayersAdded;
    };

    // left branch (branch 0)
    insertAddLayer(input0);
    for (int i = 1; i < nLayers; ++i) {
      insertAddLayer(allSigActIds.back());
    }
    TensorId actFinal0 = allSigActIds.back();

    // right branch (branch 1)
    insertAddLayer(input1);
    for (int i = 1; i < nLayers; ++i) {
      insertAddLayer(allSigActIds.back());
    }
    TensorId actFinal1 = allSigActIds.back();

    // sum of the 2 branch outputs
    auto actFinal = aiOnnx.add({actFinal0, actFinal1}, "finalAct");

    builder->addOutputTensor(actFinal);
    auto proto    = builder->getModelProto();
    auto dataFlow = DataFlow(batchesPerStep, {});

    float learnRate = 0.005;
    auto optimizer  = ConstSGD(learnRate);

    float lambda = 0.1;
    auto loss    = std::unique_ptr<Loss>(
        new L1Loss(actFinal, "l1LossVal", lambda, ReductionType::SUM));

    SessionOptions userOptions;
    std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};

    if (continuous == true) {
      userOptions.virtualGraphMode = VirtualGraphMode::Auto;
      userOptions.enablePipelining = true;
      deviceOpts["numIPUs"]        = "3";
    }

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

    if (continuous) {
      auto opSchedule = session->ir.getOpSchedule({});
      for (auto op : opSchedule) {
        auto stashOp = dynamic_cast<StashOp *>(op);
        if (stashOp) {
          BOOST_CHECK(stashOp->scheduledPreLoss == ScheduledPreLoss::Yes);
        }

        auto restoreOp = dynamic_cast<RestoreOp *>(op);
        if (restoreOp) {
          // we don't check that it is scheduled post-loss, as it is the backend
          // that controls where it is scheduled
        }
      }
    }

    // The samples (same for 0 and 1)
    std::vector<float> v_input_x(stepDataElms);

    // cumulative samples (accumulated over multiple steps).
    std::map<popart::TensorId, popart::IArray &> anchors = {};

    // write initial weights to host
    session->weightsFromHost();

    for (int i = 0; i < 3; ++i) {
      std::cout << "Iteration (call to run(...)) # " << i << std::endl;

      // generate new samples
      for (int i = 0; i < samplesPerStep; ++i) {
        for (int j = 0; j < sampleElms; ++j) {
          auto stepIndex       = i * sampleElms + j;
          v_input_x[stepIndex] = 1 + 2 * fdis(eng); // in range [0, 4]
        }
      }
      popart::NDArrayWrapper<float> input_x_wrapper(v_input_x.data(),
                                                    stepDataInfo);
      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input_x_wrapper}, {input0, input_x_wrapper}};
      popart::StepIO stepio(inputs, anchors);

      // process the samples
      session->run(stepio);
    }

    // read final weights back
    session->weightsToHost();
    session->readWeights(weightsRead);
    return WeightPair(allWeights, w_readbacks);
  };

  auto exact      = getResults(false);
  auto continuous = getResults(true);

  auto printResults = [](const WeightPair &wp) {
    auto nws = wp.start.size();
    for (auto layer = 0; layer < nws; ++layer) {
      const auto &baseline = wp.start[layer];
      std::cout << "layer " << layer << std::endl;
      for (int i = 0; i < baseline.size(); ++i) {
        std::cout << wp.start[layer][i] << " --> " << wp.end[layer][i]
                  << std::endl;
      }
    }
  };

  auto getDelta = [](const std::vector<std::vector<float>> &a,
                     const std::vector<std::vector<float>> &b) {
    float delta = 0;
    for (int layer = 0; layer < a.size(); ++layer) {
      for (int i = 0; i < a[layer].size(); ++i) {
        delta += std::abs(a[layer][i] - b[layer][i]);
      }
    }
    return delta;
  };

  if (printStdOut) {
    std::cout << "Exact: " << std::endl;
    printResults(exact);
    std::cout << "Continuous: " << std::endl;
    printResults(continuous);
  }

  auto delta_exact      = getDelta(exact.start, exact.end);
  auto delta_continuous = getDelta(continuous.start, continuous.end);
  auto delta_starts     = getDelta(continuous.start, exact.start);
  auto delta_ends       = getDelta(continuous.end, exact.end);

  // Obtained on August 1 2019 on IPUModel:
  // --------------------------------------
  // 159: delta exact 2.15162
  // 159: delta continuous 2.15149
  // 159: delta starts 0
  // 159: delta ends 0.000195466

  if (printStdOut) {
    std::cout << "delta exact " << delta_exact << std::endl;
    std::cout << "delta continuous " << delta_continuous << std::endl;
    std::cout << "delta starts " << delta_starts << std::endl;
    std::cout << "delta ends " << delta_ends << std::endl;
  }

  BOOST_CHECK(delta_ends < 0.0003);
}
