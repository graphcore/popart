// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

// We set weights as anchors, and verify that the returned values are sensible,
// agree with calls to readWeights()
//
BOOST_AUTO_TEST_CASE(WeightAnchorTest0) {

  using namespace popart;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  // accumulationFactor = replicationFactor = 1 in this example, so
  // batchSize = microBatchSize

  // samples in batch  (samples per weight update)
  int64_t batchSize = 4;

  // batches in a step (processed with each call to run(...))
  int64_t batchesPerStep = 100;

  // the number of weight update steps is batchesPerStep = 100
  // the number of samples in a step is batchesPerStep * batchSize = 400

  std::vector<int64_t> sampleShape{1, 1};

  // in this test, weights are of the same shape as data samples
  std::vector<int64_t> weightShape = sampleShape;

  // shape of a batch
  std::vector<int64_t> batchShape{batchSize, 1, 1};
  std::vector<int64_t> stepDataShape{batchesPerStep, batchSize, 1, 1};

  // When  weights are anchors (of ReturnType ALL), the the number of weights
  // that are returned will be batchesPerStep * accumulationFactor (probably,
  // see T10052)
  std::vector<int64_t> stepWeightShape{batchesPerStep, 1, 1};

  TensorInfo sampleInfo{"FLOAT", sampleShape};
  TensorInfo weightInfo = sampleInfo;
  TensorInfo batchInfo{"FLOAT", batchShape};
  TensorInfo stepDataInfo{"FLOAT", stepDataShape};
  TensorInfo stepWeightInfo{"FLOAT", stepWeightShape};

  int64_t sampleElms{1};
  int64_t batchElms      = sampleElms * batchSize;
  int64_t stepWeightElms = sampleElms * batchesPerStep;
  int64_t stepDataElms   = batchElms * batchesPerStep;

  auto input1 = builder->addInputTensor(batchInfo, "tupni");

  //
  // The model : A series of additions of a Weight, and scaling by 0.5
  //
  std::vector<float> w0Vals(sampleElms, 1.0f);
  ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
  auto w0              = builder->addInitializedInputTensor(w0Data);
  auto act0            = aiOnnx.add({w0, input1}, "act0");
  act0                 = aiGraphcore.scale({act0}, 0.5);

  std::vector<float> w1Vals(sampleElms, 1.0f);
  ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
  auto w1              = builder->addInitializedInputTensor(w1Data);
  auto act1            = aiOnnx.add({w1, act0}, "act1");
  act1                 = aiGraphcore.scale({act1}, 0.5);

  std::vector<float> w2Vals(sampleElms, 1.0f);
  ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
  auto w2              = builder->addInitializedInputTensor(w2Data);
  auto act2            = aiOnnx.add({w2, act1}, "act2");
  act2                 = aiGraphcore.scale({act2}, 0.5);

  std::vector<float> w3Vals(sampleElms, 1.0f);
  ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
  auto w3              = builder->addInitializedInputTensor(w3Data);
  auto act3            = aiOnnx.add({w3, act2}, "act3");
  act3                 = aiGraphcore.scale({act3}, 0.5);

  std::vector<float> w4Vals(sampleElms, 1.0f);
  ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
  auto w4              = builder->addInitializedInputTensor(w4Data);
  auto act4            = aiOnnx.add({w4, act3}, "act4");
  act4                 = aiGraphcore.scale({act4}, 0.5);

  std::vector<float> w5Vals(sampleElms, 1.0f);
  ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
  auto w5              = builder->addInitializedInputTensor(w5Data);
  auto act5            = aiOnnx.add({w5, act4}, "act5");
  act5                 = aiGraphcore.scale({act5}, 0.5);

  builder->addOutputTensor(act5);

  auto proto = builder->getModelProto();

  // Setting anchors as act5 and a bunch of weights
  //
  auto dataFlow = DataFlow(batchesPerStep,
                           {{act5, AnchorReturnType("ALL")},
                            {w1, AnchorReturnType("ALL")},
                            {w2, AnchorReturnType("ALL")},
                            {w3, AnchorReturnType("ALL")},
                            {w4, AnchorReturnType("ALL")},
                            {w5, AnchorReturnType("ALL")}});

  // shard over 3 IPUs
  //
  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Auto;

  auto optimizer = ConstSGD(0.01);

  auto loss = std::unique_ptr<Loss>(
      new L1Loss(act5, "l1LossVal", 0.1, ReductionType::SUM));

  auto device = createTestDevice(TEST_TARGET, 3);

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

  // All 400 samples are [[1.0f]]
  std::vector<float> v_input_0(stepDataElms, 1.0f);
  popart::NDArrayWrapper<float> input1_wrapper(v_input_0.data(), stepDataInfo);
  std::map<popart::TensorId, popart::IArray &> inputs = {
      {input1, input1_wrapper}};

  // Preparing the anchors.
  std::vector<float> v_act5_out(stepDataElms);
  popart::NDArrayWrapper<float> act5_wrapper(v_act5_out.data(), stepDataShape);

  // Note the size of the buffer we'll write to for weight anchors is
  // independent of batch-size. It is number of micro-batches (= number of
  // batches) * weight size (see T10052)
  std::vector<float> v_w1_out(stepWeightElms, -77.0f);
  popart::NDArrayWrapper<float> w1_wrapper(v_w1_out.data(), stepWeightShape);

  std::vector<float> v_w2_out(stepWeightElms, -77.0f);
  popart::NDArrayWrapper<float> w2_wrapper(v_w2_out.data(), stepWeightShape);

  std::vector<float> v_w3_out(stepWeightElms, -77.0f);
  popart::NDArrayWrapper<float> w3_wrapper(v_w3_out.data(), stepWeightShape);

  std::vector<float> v_w4_out(stepWeightElms, -77.0f);
  popart::NDArrayWrapper<float> w4_wrapper(v_w4_out.data(), stepWeightShape);

  std::vector<float> v_w5_out(stepWeightElms, -77.0f);
  popart::NDArrayWrapper<float> w5_wrapper(v_w5_out.data(), stepWeightShape);

  std::map<popart::TensorId, popart::IArray &> anchors = {{act5, act5_wrapper},
                                                          {w1, w1_wrapper},
                                                          {w2, w2_wrapper},
                                                          {w3, w3_wrapper},
                                                          {w4, w4_wrapper},
                                                          {w5, w5_wrapper}};

  WeightsIO weightsRead;
  std::vector<float> w0_readback(weightInfo.nelms(), -99.0f);
  std::vector<float> w1_readback(weightInfo.nelms(), -99.0f);
  std::vector<float> w2_readback(weightInfo.nelms(), -99.0f);
  std::vector<float> w3_readback(weightInfo.nelms(), -99.0f);
  std::vector<float> w4_readback(weightInfo.nelms(), -99.0f);
  std::vector<float> w5_readback(weightInfo.nelms(), -99.0f);
  weightsRead.insert(w0, {w0_readback.data(), weightInfo});
  weightsRead.insert(w1, {w1_readback.data(), weightInfo});
  weightsRead.insert(w2, {w2_readback.data(), weightInfo});
  weightsRead.insert(w3, {w3_readback.data(), weightInfo});
  weightsRead.insert(w4, {w4_readback.data(), weightInfo});
  weightsRead.insert(w5, {w5_readback.data(), weightInfo});

  // write initial weights to host
  session->weightsFromHost();

  // read final weights back
  session->weightsToHost();
  session->readWeights(weightsRead);

  // write followed by immediate read : read back initializer values
  std::cout << "Testing that immediately read back initializer vals are correct"
            << std::endl;
  BOOST_CHECK(w0_readback.back() == w0Vals.back());
  BOOST_CHECK(w1_readback.back() == w1Vals.back());
  BOOST_CHECK(w2_readback.back() == w2Vals.back());
  BOOST_CHECK(w3_readback.back() == w3Vals.back());
  BOOST_CHECK(w4_readback.back() == w4Vals.back());
  BOOST_CHECK(w5_readback.back() == w5Vals.back());

  for (int i = 0; i < 3; ++i) {
    std::cout << "Iteration (call to run(...)) # " << i << std::endl;

    // we will use the same data (inputs) for every step
    popart::StepIO stepio(inputs, anchors);

    // process the 400 samples (100 batches), streaming back anchors each batch
    session->run(stepio);

    // read final weights back
    session->weightsToHost();
    session->readWeights(weightsRead);

    std::cout << "Testing that final (weight) anchors are the same as values "
                 "from readWeights()"
              << std::endl;

    BOOST_CHECK(w1_readback.back() == v_w1_out.back());
    BOOST_CHECK(w2_readback.back() == v_w2_out.back());
    BOOST_CHECK(w3_readback.back() == v_w3_out.back());
    BOOST_CHECK(w4_readback.back() == v_w4_out.back());
    BOOST_CHECK(w5_readback.back() == v_w5_out.back());

    std::cout << "all the w1 anchor returns" << std::endl;
    for (auto x : v_w1_out) {
      std::cout << x << "  ";
    }
    std::cout << std::endl;
  }
}
