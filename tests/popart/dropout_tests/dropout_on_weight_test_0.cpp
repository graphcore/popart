// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PipelineTrainingTest0

#include <../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

// Dropout is usually performed on an activation Tensor. This test checks that
// it works on a Var Tensor too.

BOOST_AUTO_TEST_CASE(AutoVirtualGraphReluOnWeightTest0) {

  enum class TestType {
    PipelineHardware,
    PipelineHardwareRecomputation,
    SingleBatchSimulator
  };

  auto test = [](TestType tt) {
    using namespace popart;

    // weights will be initiliased randomly
    int seed = 1011;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(0.f, 1.f);

    // --------- Defining Tensor sizes ----------
    // in this example, accumulationFactor = replicationFactor = 1,
    // so batchSize = microBatchSize
    // number of samples in a batch ( = samples per weight update)
    int64_t batchSize = 11;
    // the number of weight update steps is batchesPerStep,
    int64_t batchesPerStep = (tt != TestType::SingleBatchSimulator ? 13 : 1);
    // samples in a step (samples processed with each SingleBatchSimulator to
    // run(...))
    int64_t samplesPerStep = batchesPerStep * batchSize;
    // an input data sample will be a rank-1 tensor of size,
    int64_t seqLen = 7;
    std::vector<int64_t> sampleShape{seqLen};
    // in this test, weights are of the same shape as data samples,
    std::vector<int64_t> weightShape = sampleShape;
    // shape of a batch
    std::vector<int64_t> batchShape{batchSize, seqLen};
    std::vector<int64_t> stepDataShape{batchesPerStep, batchSize, seqLen};
    std::vector<int64_t> stepWeightShape{batchesPerStep, seqLen};
    TensorInfo sampleInfo{"FLOAT", sampleShape};
    TensorInfo weightInfo = sampleInfo;
    TensorInfo batchInfo{"FLOAT", batchShape};
    TensorInfo stepDataInfo{"FLOAT", stepDataShape};
    int64_t weightsElms    = seqLen;
    int64_t batchElms      = seqLen * batchSize;
    int64_t stepDataElms   = batchElms * batchesPerStep;
    int64_t stepWeightElms = batchesPerStep * weightsElms;

    // ------- Building the Model -----------------
    // A layer of the network:
    //
    // activation
    //   input
    //    |
    //    |    weight
    //    |     /
    //    |  dropout
    //    |   /
    //    |  /
    //    add
    //     |
    //     |
    //     |
    // activation
    //    out
    //

    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    auto input1      = builder->addInputTensor(batchInfo, "1tupni");
    // weights by layer
    std::map<int, std::vector<float>> wVals;
    // name of weights tensor by layer:
    std::map<int, TensorId> wNames;

    // name of mask tensor by layer:
    std::map<int, TensorId> mNames;

    float pDrop = 1. / 3.0f;

    // initial weight magnitude (sign will be chosen randomly)
    float wMag = 1000.;
    std::vector<TensorId> maskIds;
    auto addLayer = [&aiOnnx,
                     pDrop,
                     &maskIds,
                     &builder,
                     sampleInfo,
                     &wVals,
                     &eng,
                     &fdis,
                     seqLen,
                     &wNames,
                     &mNames,
                     wMag](TensorId actIn, int layerNumber) {
      std::vector<float> wVec(seqLen);
      for (int j = 0; j < seqLen; ++j) {
        wVec[j] = fdis(eng) > 0.5 ? -wMag : wMag;
      }
      wVals.insert({layerNumber, std::move(wVec)});
      ConstVoidData wData{wVals.at(layerNumber).data(), sampleInfo};
      auto w = builder->addInitializedInputTensor(wData);
      wNames.insert({layerNumber, w});
      auto w0_drop_pair = aiOnnx.dropout({w}, 2, pDrop);
      auto w_pos        = w0_drop_pair[0];
      maskIds.push_back(w0_drop_pair[1]);
      mNames.insert({layerNumber, w0_drop_pair[1]});
      auto actOut = aiOnnx.add({w_pos, actIn}, "act0");
      return actOut;
    };

    auto actOut = addLayer(input1, 0);
    for (int i = 1; i < 7; ++i) {
      actOut = addLayer(actOut, i);
    }
    float lambda = 1;
    actOut       = builder->aiGraphcoreOpset1().l1loss(
        {actOut}, lambda, ReductionType::Sum);
    auto proto = builder->getModelProto();

    // -------------- Losses, anchors, etc ----------------------
    AnchorReturnTypeMap anchorMap;
    for (auto &maskId : maskIds) {
      anchorMap.insert({maskId, AnchorReturnType("All")});
    }
    auto dataFlow   = DataFlow(batchesPerStep, anchorMap);
    float learnRate = 1;
    auto optimizer  = ConstSGD(learnRate);

    auto device = createTestDevice(TEST_TARGET, 2);

    SessionOptions userOptions;
    userOptions.virtualGraphMode = VirtualGraphMode::Auto;
    userOptions.enablePipelining =
        (tt == TestType::PipelineHardware ? true : false);
    userOptions.autoRecomputation =
        (tt == TestType::PipelineHardwareRecomputation
             ? RecomputationType::Standard
             : RecomputationType::None);

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        actOut,
        optimizer,
        device,
        InputShapeInfo(),
        userOptions,
        popart::Patterns(PatternsLevel::Default));

    session->prepareDevice();

    // The samples, all with LARGE POSITIVE value
    std::vector<float> v_input_0(stepDataElms, wMag * 100.);

    using BoolType = uint8_t;
    std::map<TensorId, std::vector<BoolType>> maskAnchorReturns;
    std::map<TensorId, popart::NDArrayWrapper<BoolType>> ancWrappers;
    std::map<TensorId, popart::IArray &> anchors;
    for (TensorId maskId : maskIds) {
      maskAnchorReturns.emplace(maskId, std::vector<BoolType>(stepWeightElms));

      NDArrayWrapper<BoolType> ancWrapper(maskAnchorReturns[maskId].data(),
                                          stepWeightShape);

      ancWrappers.emplace(
          maskId,
          popart::NDArrayWrapper<BoolType>{maskAnchorReturns[maskId].data(),
                                           stepWeightShape});
    }

    for (TensorId maskId : maskIds) {
      anchors.insert({maskId, ancWrappers.at(maskId)});
    }

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

    int nSteps    = 5;
    auto nUpdates = samplesPerStep * nSteps;
    for (int i = 0; i < nSteps; ++i) {
      std::cout << "Iteration (call to run(...)) # " << i << std::endl;

      popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                   stepDataInfo);

      std::map<popart::TensorId, popart::IArray &> inputs = {
          {input1, input1_wrapper}};

      popart::StepIO stepio(inputs, anchors);

      // We fix the seed, so we expect the set of masks to be the same step vs
      // step. However, inside a step, batch vs batch, you will get different
      // masks.
      session->setRandomSeed(31415);

      // process the 400 samples (100 batches), streaming back mask anchors each
      // batch
      session->run(stepio);
    }

    // read final weights back
    session->weightsToHost();
    session->readWeights(weightsRead);

    // Check that the dropout masks are different on hardware.
    auto maskZero = maskAnchorReturns.at(mNames.at(0));
    for (auto &x : weightsOut) {
      auto layer = x.first;
      auto wName = wNames.at(layer);
      auto mName = mNames.at(layer);

      auto &returnedWeights = x.second;
      auto mask             = maskAnchorReturns.at(mName);
      auto &initialWeights  = wVals.at(layer);

      std::vector<uint32_t> maskSum(seqLen, 0);

      // Combine all masks into one. mask is of size stepWeightElms, but is
      // effectively a concatenation of of all masks for this layer for all
      // batches. On hardware these will be different, as expected.
      for (int i = 0; i < mask.size(); i++) {
        maskSum[i % seqLen] += static_cast<uint32_t>(mask.at(i));
      }
      for (int i = 0; i < returnedWeights.size(); ++i) {
        auto mval = static_cast<uint32_t>(maskSum.at(i));
        // Dividing by 1 - pDrop causes too many floating point rounding
        // errors, so just multiply by 1 / (1 - 1/3) = 1.5
        float delta = mval * learnRate * lambda * batchSize * nSteps * 1.5f;
        // This is a percentage difference test, i.e. values within 1e-8
        // There is a small difference due to FP rounding errors.
        BOOST_CHECK_CLOSE(static_cast<double>(returnedWeights.at(i) + delta),
                          static_cast<double>(initialWeights.at(i)),
                          1e-6);
      }
    }
  };

  test(TestType::PipelineHardware);

  test(TestType::PipelineHardwareRecomputation);

  test(TestType::SingleBatchSimulator);
}
