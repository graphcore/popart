// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Train0TopkTest

#include <../random_util.hpp>
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
#include <tuple>
#include <vector>

BOOST_AUTO_TEST_CASE(Train0TopK) {

  // input tensor X
  // The input tensor will be rank 3, D0 x D1 x D2 x D3
  int D0 = 3;
  int D1 = 7;
  int D2 = 2;
  int D3 = 5;

  // We will perform top-k on axis "axis = 1" and so the output will
  // be of size D0 x top_k x D2 x D3, where top_k <= D1.
  int axis = 1;
  std::vector<int> top_ks{1, 3, D1};

  auto test = [D0, D1, D2, D3, axis](int top_k) {
    // basic differentiation shows that we expect
    // d_topk_out = 2*scale*lossLambda*topk_out

    using namespace popart;

    // we will generate random input data
    int seed = 1013;
    DefaultRandomEngine eng(seed);
    UniformRealDistribution<float> fdis(-4.f, +4.f);

    // prepare to build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    TensorInfo xInfo{"FLOAT", std::vector<int64_t>{D0, D1, D2, D3}};
    TensorId xId = builder->addInputTensor(xInfo);
    std::vector<float> vXData(xInfo.nelms());
    for (auto &val : vXData) {
      val = fdis(eng);
    }

    // the networt scale((topk)**2) with loss L1.
    float scaleFactor = 3.0f;
    float lossLambda  = 0.26;

    Shape outShape = xInfo.shape();
    outShape[axis] = top_k;
    TensorInfo outValuesInfo{xInfo.dataType(), outShape};

    // Prepare the baseline data, perform D0*D2
    // sorts of vectors of size D1 to get the values and the indices
    std::vector<int> expectedOutputIndices(outValuesInfo.nelms(), -1);
    std::vector<float> expectedOutputValues(outValuesInfo.nelms(), -1.0f);
    std::vector<float> expectedInputGradients(xInfo.nelms(), 0.0f);
    int stride0 = D1 * D2 * D3;
    int stride1 = D2 * D3;
    int stride2 = D3;

    int outStride0 = top_k * D2 * D3;
    int outStride1 = D2 * D3;
    int outStride2 = D3;

    for (int d0 = 0; d0 < D0; ++d0) {
      for (int d2 = 0; d2 < D2; ++d2) {
        for (int d3 = 0; d3 < D3; ++d3) {

          std::vector<std::tuple<float, int>> toSort(D1);
          for (int d1 = 0; d1 < D1; ++d1) {
            toSort[d1] = std::tuple<float, int>(
                vXData[d0 * stride0 + d1 * stride1 + d2 * stride2 + d3], d1);
          }

          // sort, largest to smallest
          std::sort(toSort.rbegin(), toSort.rend());

          // populate expected values
          for (int d1 = 0; d1 < top_k; ++d1) {
            int sortIndex = std::get<1>(toSort[d1]);
            auto sortVal  = std::get<0>(toSort[d1]);

            int index =
                d0 * outStride0 + d1 * outStride1 + d2 * outStride2 + d3;
            expectedOutputIndices[index] = sortIndex;
            expectedOutputValues[index]  = sortVal;
            expectedInputGradients[d0 * stride0 + sortIndex * stride1 +
                                   d2 * stride2 + d3] =
                2 * scaleFactor * lossLambda * std::get<0>(toSort[d1]);
          }
        }
      }
    }

    auto topkOut = aiOnnx.topk({xId}, top_k, axis);

    auto values  = topkOut[0];
    auto indices = topkOut[1];

    auto squaredOut = aiOnnx.mul({values, values});
    auto halvedOut  = aiGraphcore.scale({squaredOut}, scaleFactor);

    builder->addOutputTensor(halvedOut);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // create the IR
    auto art      = AnchorReturnType("All");
    auto dataFlow = DataFlow(1, {{reservedGradientPrefix() + xId, art}});

    auto device = popart::createTestDevice(TEST_TARGET);

    auto opts = SessionOptions();

    float learnRate = 0.1;
    auto optimizer  = ConstSGD(learnRate);
    std::vector<Loss *> losses{
        new L1Loss(halvedOut, "l1LossVal", lossLambda, ReductionType::Sum)};

    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        losses,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));

    // prepare the anchors. We test just the
    // gradient of output values.
    std::vector<float> rawXGrad(xInfo.nelms());
    popart::NDArrayWrapper<float> xGrad(rawXGrad.data(), xInfo.shape());

    std::map<popart::TensorId, popart::IArray &> anchors = {
        {reservedGradientPrefix() + xId, xGrad}};

    session->prepareDevice();

    popart::NDArrayWrapper<float> xData(vXData.data(), xInfo);

    std::map<popart::TensorId, popart::IArray &> inputs = {{xId, xData}};

    popart::StepIO stepio(inputs, anchors);

    session->run(stepio);

    // confirm the gradient values agree (exactly...)
    BOOST_CHECK_EQUAL_COLLECTIONS(rawXGrad.begin(),
                                  rawXGrad.end(),
                                  expectedInputGradients.begin(),
                                  expectedInputGradients.end());
  };

  for (int top_k : top_ks) {
    test(top_k);
  }
}
