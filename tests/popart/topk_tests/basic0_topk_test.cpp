// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Basic0TopkTest

#include <../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/testdevice.hpp>

#include <vector>

BOOST_AUTO_TEST_CASE(Basic0TopK_Opset9) {

  using namespace popart;

  // generate random input data
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare to build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  // input tensor X
  // The input tensor will be rank 3, D0 x D1 x D2
  int D0 = 3;
  int D1 = 11;
  int D2 = 7;
  TensorInfo xInfo{"FLOAT", std::vector<int64_t>{D0, D1, D2}};
  TensorId xId = builder->addInputTensor(xInfo);
  std::vector<float> vXData(xInfo.nelms());
  for (auto &val : vXData) {
    val = fdis(eng);
  }

  // We will perform top-k on axis "axis" and so the output will
  // be of size D0 x top_k x D2, where top_k <= D1.
  int axis  = 1;
  int top_k = 4;

  Shape outShape = xInfo.shape();
  outShape[axis] = top_k;
  TensorInfo outValuesInfo{xInfo.dataType(), outShape};

  // Prepare the baseline data, perform D0*D2 sorts of vectors of size D1
  std::vector<int> expectedOutputIndices(outValuesInfo.nelms(), -1);
  std::vector<float> expectedOutputValues(outValuesInfo.nelms(), -1.0f);
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D2; ++j) {
      std::vector<std::tuple<float, int>> oneD2Sort(D1);
      for (int k_ = 0; k_ < D1; ++k_) {
        oneD2Sort[k_] =
            std::tuple<float, int>(vXData[i * D1 * D2 + D2 * k_ + j], k_);
      }
      // sort largest to smallest
      std::sort(oneD2Sort.rbegin(), oneD2Sort.rend());
      for (int k_ = 0; k_ < top_k; ++k_) {
        expectedOutputIndices[i * top_k * D2 + D2 * k_ + j] =
            std::get<1>(oneD2Sort[k_]);
        expectedOutputValues[i * top_k * D2 + D2 * k_ + j] =
            std::get<0>(oneD2Sort[k_]);
      }
    }
  }

  auto topkOut = aiOnnx.topk({xId}, top_k, axis);

  // Top-K has 2 oututs, the indices and values, both of size D0 x top_k x D1
  BOOST_CHECK(topkOut.size() == 2);
  auto values  = topkOut[0];
  auto indices = topkOut[1];

  builder->addOutputTensor(values);
  builder->addOutputTensor(indices);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // create the IR
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{values, art}, {indices, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts = SessionOptions();

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::NoPatterns).enableRuntimeAsserts(false));

  // prepare the anchors
  std::vector<float> rawOutputValues(outValuesInfo.nelms());
  std::vector<int> rawOutputIndices(outValuesInfo.nelms());
  popart::NDArrayWrapper<float> outValues(rawOutputValues.data(), outShape);
  popart::NDArrayWrapper<int> outIndices(rawOutputIndices.data(), outShape);

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {values, outValues}, {indices, outIndices}};

  session->prepareDevice();

  popart::NDArrayWrapper<float> xData(vXData.data(), xInfo);
  std::map<popart::TensorId, popart::IArray &> inputs = {{xId, xData}};

  popart::StepIO stepio(inputs, anchors);

  session->run(stepio);

  // confirm the indices agree
  BOOST_CHECK_EQUAL_COLLECTIONS(rawOutputIndices.begin(),
                                rawOutputIndices.end(),
                                expectedOutputIndices.begin(),
                                expectedOutputIndices.end());
  // confirm the values agree
  float sumAbsDiff{0.0f};
  for (int i = 0; i < expectedOutputIndices.size(); ++i) {
    sumAbsDiff += std::abs(rawOutputValues[i] - expectedOutputValues[i]);
  }
  BOOST_CHECK(sumAbsDiff < 1e-9f);
}

BOOST_AUTO_TEST_CASE(Basic0TopK_Opset10) {

  using namespace popart;

  // generate random input data
  int seed = 1013;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-4.f, +4.f);

  // prepare to build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset10();

  // input tensor X
  // The input tensor will be rank 3, D0 x D1 x D2
  int D0 = 3;
  int D1 = 11;
  int D2 = 7;
  TensorInfo xInfo{"FLOAT", std::vector<int64_t>{D0, D1, D2}};
  TensorId xId = builder->addInputTensor(xInfo);

  int top_k = 4;
  TensorInfo kShape{"INT64", std::vector<int64_t>{1}};
  int64_t kVals[1]    = {top_k};
  ConstVoidData kData = {kVals, kShape};

  std::vector<float> vXData(xInfo.nelms());
  for (auto &val : vXData) {
    val = fdis(eng);
  }

  // We will perform top-k on axis "axis" and so the output will
  // be of size D0 x top_k x D2, where top_k <= D1.
  int axis = 1;

  Shape outShape = xInfo.shape();
  outShape[axis] = top_k;
  TensorInfo outValuesInfo{xInfo.dataType(), outShape};

  // Prepare the baseline data, perform D0*D2 sorts of vectors of size D1
  std::vector<int> expectedOutputIndices(outValuesInfo.nelms(), -1);
  std::vector<float> expectedOutputValues(outValuesInfo.nelms(), -1.0f);
  for (int i = 0; i < D0; ++i) {
    for (int j = 0; j < D2; ++j) {
      std::vector<std::tuple<float, int>> oneD2Sort(D1);
      for (int k_ = 0; k_ < D1; ++k_) {
        oneD2Sort[k_] =
            std::tuple<float, int>(vXData[i * D1 * D2 + D2 * k_ + j], k_);
      }
      // sort largest to smallest
      std::sort(oneD2Sort.rbegin(), oneD2Sort.rend());
      for (int k_ = 0; k_ < top_k; ++k_) {
        expectedOutputIndices[i * top_k * D2 + D2 * k_ + j] =
            std::get<1>(oneD2Sort[k_]);
        expectedOutputValues[i * top_k * D2 + D2 * k_ + j] =
            std::get<0>(oneD2Sort[k_]);
      }
    }
  }

  auto kId     = aiOnnx.constant(kData);
  auto topkOut = aiOnnx.topk({xId, kId}, axis);

  // Top-K has 2 oututs, the indices and values, both of size D0 x top_k x D1
  BOOST_CHECK(topkOut.size() == 2);
  auto values  = topkOut[0];
  auto indices = topkOut[1];

  builder->addOutputTensor(values);
  builder->addOutputTensor(indices);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // create the IR
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{values, art}, {indices, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts = SessionOptions();

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      opts,
      popart::Patterns(PatternsLevel::NoPatterns).enableRuntimeAsserts(false));

  // prepare the anchors
  std::vector<float> rawOutputValues(outValuesInfo.nelms());
  std::vector<int> rawOutputIndices(outValuesInfo.nelms());
  popart::NDArrayWrapper<float> outValues(rawOutputValues.data(), outShape);
  popart::NDArrayWrapper<int> outIndices(rawOutputIndices.data(), outShape);

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {values, outValues}, {indices, outIndices}};

  session->prepareDevice();

  popart::NDArrayWrapper<float> xData(vXData.data(), xInfo);
  std::map<popart::TensorId, popart::IArray &> inputs = {{xId, xData}};

  popart::StepIO stepio(inputs, anchors);

  session->run(stepio);

  // confirm the indices agree
  BOOST_CHECK_EQUAL_COLLECTIONS(rawOutputIndices.begin(),
                                rawOutputIndices.end(),
                                expectedOutputIndices.begin(),
                                expectedOutputIndices.end());
  // confirm the values agree
  float sumAbsDiff{0.0f};
  for (int i = 0; i < expectedOutputIndices.size(); ++i) {
    sumAbsDiff += std::abs(rawOutputValues[i] - expectedOutputValues[i]);
  }
  BOOST_CHECK(sumAbsDiff < 1e-9f);
}
