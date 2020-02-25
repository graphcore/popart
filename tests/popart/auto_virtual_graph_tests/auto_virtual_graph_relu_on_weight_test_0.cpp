#define BOOST_TEST_MODULE PipelineTrainingTest0

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

BOOST_AUTO_TEST_CASE(AutoVirtualGraphReluOnWeightTest0) {

  using namespace popart;

  // weights will be initiliased randomly
  int seed = 1011;
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<float> fdis(0, 1);

  // --------- Defining Tensor sizes ----------
  //

  // in this example, accumulationFactor = replicationFactor = 1,
  // so batchSize = microBatchSize

  // number of samples in a batch ( = samples per weight update)
  int64_t batchSize = 10;

  // the number of weight update steps is batchesPerStep:
  int64_t batchesPerStep = 10;

  // samples in a step (samples processed with each call to run(...))
  int64_t samplesPerStep = batchesPerStep * batchSize;

  // an input data sample will be a rank-1 tensor if size:
  int64_t seqLen = 10;
  std::vector<int64_t> sampleShape{seqLen};

  // in this test, weights are of the same shape as data samples
  std::vector<int64_t> weightShape = sampleShape;

  // shape of a batch
  std::vector<int64_t> batchShape{batchSize, seqLen};
  std::vector<int64_t> stepDataShape{batchesPerStep, batchSize, seqLen};

  TensorInfo sampleInfo{"FLOAT", sampleShape};
  TensorInfo weightInfo = sampleInfo;
  TensorInfo batchInfo{"FLOAT", batchShape};
  TensorInfo stepDataInfo{"FLOAT", stepDataShape};

  int64_t batchElms    = seqLen * batchSize;
  int64_t stepDataElms = batchElms * batchesPerStep;

  // ------- Building the Model -----------------
  //
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  auto input1 = builder->addInputTensor(batchInfo, "1tupni");

  // weights by layer
  std::map<int, std::vector<float>> wVals;
  // name of weights tensor by layer:
  std::map<int, TensorId> wNames;

  // A layer of the network:
  //
  // activation
  //   input
  //    |
  //    |    weight
  //    |     /
  //    |  relu
  //    |   /
  //    |  /
  //    add
  //     |
  //     |
  //     |
  // activation
  //    out
  //

  // initial weight magnitude (sign chosen randomly)
  float wMag = 1000.;

  auto addLayer = [&aiOnnx,
                   &builder,
                   sampleInfo,
                   &wVals,
                   &eng,
                   &fdis,
                   seqLen,
                   &wNames,
                   wMag](TensorId actIn, int layerNumber) {
    std::vector<float> wVec(seqLen);
    for (int j = 0; j < seqLen; ++j) {
      wVec[j] = fdis(eng) > 0.5 ? -wMag : wMag;
    }
    wVals.insert({layerNumber, std::move(wVec)});

    ConstVoidData wData{wVals.at(layerNumber).data(), sampleInfo};
    auto w = builder->addInitializedInputTensor(wData);
    wNames.insert({layerNumber, w});
    auto w_pos  = aiOnnx.relu({w});
    auto actOut = aiOnnx.add({w_pos, actIn}, "act0");
    return actOut;
  };

  auto actOut = addLayer(input1, 0);
  actOut      = addLayer(actOut, 1);
  actOut      = addLayer(actOut, 2);
  actOut      = addLayer(actOut, 3);
  actOut      = addLayer(actOut, 4);
  actOut      = addLayer(actOut, 5);
  actOut      = addLayer(actOut, 6);

  builder->addOutputTensor(actOut);

  auto proto = builder->getModelProto();

  // -------------- Losses, anchors, etc ----------------------
  //
  //

  // No anchors
  auto dataFlow = DataFlow(batchesPerStep, {});

  float learnRate = 1;
  auto optimizer  = ConstSGD(learnRate);

  float lambda = 1;
  auto loss    = std::unique_ptr<Loss>(
      new L1Loss(actOut, "l1LossVal", lambda, ReductionType::SUM));

  auto device = createTestDevice(TEST_TARGET, 3);

  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Auto;

  // TODO : extend to pipelining T10204
  userOptions.enablePipelining = false;

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

  // The 400 samples, all with LARGE POSITIVE value
  std::vector<float> v_input_0(stepDataElms, wMag * 100.);
  std::map<popart::TensorId, popart::IArray &> anchors = {};

  WeightsIO weightsRead;
  std::map<int, std::vector<float>> weightsOut;
  for (auto &x : wVals) {
    int layer = x.first;
    weightsOut.insert({layer, std::vector<float>(x.second.size(), -99)});
    weightsRead.insert(wNames.at(layer),
                       {weightsOut.at(layer).data(), weightInfo});
  }

  // write initial weights to host
  session->weightsFromHost();

  int nSteps = 3;
  for (int i = 0; i < nSteps; ++i) {
    std::cout << "Iteration (call to run(...)) # " << i << std::endl;

    popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                 stepDataInfo);

    std::map<popart::TensorId, popart::IArray &> inputs = {
        {input1, input1_wrapper}};

    popart::StepIO stepio(inputs, anchors);

    // process the 400 samples (100 batches), streaming back anchors each batch
    session->run(stepio);
  }

  // read final weights back
  session->weightsToHost();
  session->readWeights(weightsRead);

  // Now testing that it trained correctly
  // -------------------------------
  //
  // if the initial weight was negative, the gradient won't get through the
  // relu, so the weight will be unchanged
  //
  // if the initial weight was positive, the gradient WILL get through the relu.
  // The gradient is always negative, as the input values are much larger than
  // any weights. So the final weight should be nSteps * lambda * learnRate *
  // samplesPerStep smaller than the initial weight
  //
  for (auto &x : weightsOut) {
    auto layer            = x.first;
    auto &returnedWeights = x.second;
    auto &initialWeights  = wVals.at(layer);
    for (int i = 0; i < returnedWeights.size(); ++i) {
      float diff;
      if (initialWeights.at(i) < 0) {
        diff = returnedWeights.at(i) - initialWeights.at(i);
      }

      else {
        diff = returnedWeights.at(i) +
               samplesPerStep * nSteps * lambda * learnRate -
               initialWeights.at(i);
      }
      BOOST_CHECK(std::abs(diff) < 1e-7);
    }
  }
}
