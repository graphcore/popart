#define BOOST_TEST_MODULE PipelineTrainingTest0

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>

#include <algorithm>
#include <map>
#include <random>
#include <tuple>
#include <vector>

//
// In this depp linear model, the continuous weight update is equivalent to
// exact SGD. This enables us to test the final weights numerically.
//
BOOST_AUTO_TEST_CASE(ContinuousEquivalentTest0) {

  int seed = 1011;
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<float> fdis(0, 1);

  using namespace poponnx;

  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();

  // accumulationFactor = replicationFactor = 1 in this example,
  // so batchSize = microBatchSize

  // samples in batch  (samples per weight update)
  int64_t batchSize = 4;

  int64_t batchesPerStep = 100;

  // samples in a step (samples processed with each call to run(...))
  int64_t samplesPerStep = batchesPerStep * batchSize;

  // the number of weight update steps is batchesPerStep = 100
  // the number of samples in a step is batchesPerStep * batchSize = 400

  int64_t sampleHeight = 8;
  std::vector<int64_t> sampleShape{sampleHeight, sampleHeight};

  // in this test, weights are of the same shape as data samples
  std::vector<int64_t> weightShape = sampleShape;

  // shape of a batch
  std::vector<int64_t> batchShape{batchSize, sampleHeight, sampleHeight};
  std::vector<int64_t> stepDataShape{
      batchesPerStep, batchSize, sampleHeight, sampleHeight};

  TensorInfo sampleInfo{"FLOAT", sampleShape};
  TensorInfo weightInfo = sampleInfo;
  TensorInfo batchInfo{"FLOAT", batchShape};
  TensorInfo stepDataInfo{"FLOAT", stepDataShape};

  int64_t sampleElms{sampleHeight * sampleHeight};
  int64_t weightsElms  = sampleElms;
  int64_t batchElms    = sampleElms * batchSize;
  int64_t stepDataElms = batchElms * batchesPerStep;

  auto input1 = builder->addInputTensor(batchInfo, "1tupni");

  // The model : A series of additions of a Weight, and occassionally scaling
  // by 1.0 = (-1*-1).
  //
  // All weights are initialized as zero, and all inputs will be large (positive
  // or negative). This means that the l1 loss gradient (+lambda or -lambda)
  // will depend exclusively on the input (we use a small enough learning rate
  // so that this fact remains true during our training steps) This means that
  // the gradient applied to weights depends only on the sample for which it is
  // computed, which ensures the equivalence between continous and gradient
  // accumuated pipelining
  //
  std::vector<float> w0Vals(sampleElms, 0.0f);
  ConstVoidData w0Data = {w0Vals.data(), sampleInfo};
  auto w0              = builder->addInitializedInputTensor(w0Data);
  auto act0            = aiOnnx.add({w0, input1}, "act0");
  act0                 = aiGraphcore.scale({act0}, -1.0);
  act0                 = aiGraphcore.scale({act0}, -1.0);

  std::vector<float> w1Vals(sampleElms, 0.0f);
  ConstVoidData w1Data = {w1Vals.data(), sampleInfo};
  auto w1              = builder->addInitializedInputTensor(w1Data);
  auto act1            = aiOnnx.add({w1, act0}, "act1");

  std::vector<float> w2Vals(sampleElms, 0.0f);
  ConstVoidData w2Data = {w2Vals.data(), sampleInfo};
  auto w2              = builder->addInitializedInputTensor(w2Data);
  auto act2            = aiOnnx.add({w2, act1}, "act2");
  act2                 = aiGraphcore.scale({act2}, -1.0);
  act2                 = aiGraphcore.scale({act2}, -1.0);

  std::vector<float> w3Vals(sampleElms, 0.0f);
  ConstVoidData w3Data = {w3Vals.data(), sampleInfo};
  auto w3              = builder->addInitializedInputTensor(w3Data);
  auto act3            = aiOnnx.add({w3, act2}, "act3");

  std::vector<float> w4Vals(sampleElms, 0.0f);
  ConstVoidData w4Data = {w4Vals.data(), sampleInfo};
  auto w4              = builder->addInitializedInputTensor(w4Data);
  auto act4            = aiOnnx.add({w4, act3}, "act4");
  act4                 = aiGraphcore.scale({act4}, -1.0);
  act4                 = aiGraphcore.scale({act4}, -1.0);

  std::vector<float> w5Vals(sampleElms, 0.0f);
  ConstVoidData w5Data = {w5Vals.data(), sampleInfo};
  auto w5              = builder->addInitializedInputTensor(w5Data);
  auto act5            = aiOnnx.add({w5, act4}, "act5");

  builder->addOutputTensor(act5);

  auto proto = builder->getModelProto();

  // Setting anchors as act5
  auto dataFlow = DataFlow(batchesPerStep, {{act5, AnchorReturnType("ALL")}});

  // shard over 3 IPUs, and enable pipelining
  SessionOptions userOptions;
  userOptions.enableVirtualGraphs = true;
  userOptions.autoVirtualGraph    = true;
  userOptions.enablePipelining    = true;
  std::map<std::string, std::string> deviceOpts{{"numIPUs", "3"}};

  float learnRate = 0.01;
  auto optimizer  = ConstSGD(learnRate);

  float lambda = 0.1;
  auto loss    = std::unique_ptr<Loss>(
      new L1Loss(act5, "l1LossVal", lambda, ReductionType::SUM));

  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  auto session = poponnx::TrainingSession::createFromOnnxModel(
      proto,
      dataFlow,
      {loss.get()},
      optimizer,
      device,
      InputShapeInfo(),
      userOptions,
      poponnx::Patterns(PatternsLevel::DEFAULT));

  session->prepareDevice();

  // The 400 samples.
  std::vector<float> v_input_0(stepDataElms);

  // cumulative samples (accumulated over multiple steps).
  std::vector<float> v_sample_sum_0(weightInfo.nelms(), 0.0f);

  // Preparing the anchors.
  std::vector<float> v_act5_out(stepDataElms);
  poponnx::NDArrayWrapper<float> act5_wrapper(v_act5_out.data(), stepDataShape);

  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
      {act5, act5_wrapper}};

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

  float sampleNumVal = 100.0f;
  for (int i = 0; i < 3; ++i) {
    std::cout << "Iteration (call to run(...)) # " << i << std::endl;

    // get new samples
    for (int i = 0; i < samplesPerStep; ++i) {
      for (int j = 0; j < sampleElms; ++j) {
        auto stepIndex       = i * sampleElms + j;
        v_input_0[stepIndex] = fdis(eng) > 0.5 ? -sampleNumVal : +sampleNumVal;
        v_sample_sum_0[j] += v_input_0[stepIndex];
      }
    }
    poponnx::NDArrayWrapper<float> input1_wrapper(v_input_0.data(),
                                                  stepDataInfo);
    std::map<poponnx::TensorId, poponnx::IArray &> inputs = {
        {input1, input1_wrapper}};
    poponnx::StepIO stepio(inputs, anchors);

    // process the 400 samples (100 batches), streaming back anchors each batch
    session->run(stepio);
  }

  // read final weights back
  session->weightsToHost();
  session->readWeights(weightsRead);

  std::vector<std::vector<float> *> ws = {&w0_readback,
                                          &w1_readback,
                                          &w2_readback,
                                          &w3_readback,
                                          &w4_readback,
                                          &w5_readback};

  // get sum of absolute differences between computed and expected
  float sumAbsDiff = 0.0;
  for (auto wv_p : ws) {
    for (int i = 0; i < w0_readback.size(); ++i) {
      sumAbsDiff += std::abs((*wv_p)[i] + v_sample_sum_0[i] * learnRate *
                                              lambda / sampleNumVal);
    }
  }
  BOOST_CHECK(sumAbsDiff < 1e-5);
}
