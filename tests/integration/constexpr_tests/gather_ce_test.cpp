// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprGatherTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

template <typename DATA_IN_TYPE, typename INDICES_TYPE>
void ConstExprTest_Gather_Type(std::string data_in_type,
                               std::string indices_type) {
  // The compute graph :
  //
  // data ---- GATHER ----> Add([0]) ----> output

  // We will gather a rank-4 tensor:
  Shape inShape = {2, 5, 3, 4};
  // Expect a rank-5 tensor result, with a smaller axis 1 and a new axis 2
  Shape expectedOutShape = {2, 2, 3, 3, 4};
  // Gather along axis 1
  int64_t axis = 1;

  TensorInfo inDataInfo{data_in_type, inShape};
  std::vector<DATA_IN_TYPE> inData(inDataInfo.nelms());
  for (int64_t i = 0; i < inDataInfo.nelms(); ++i) {
    // Fill the array with the indices as value
    inData[i] = static_cast<DATA_IN_TYPE>(i);
  }

  Shape indicesShape = {2, 3};
  TensorInfo inIndicesInfo{indices_type, indicesShape};
  std::vector<INDICES_TYPE> indices({1, 3, 0, 2, 4, 1});

  TensorInfo expectedOutdataInfo{data_in_type, expectedOutShape};
  std::vector<DATA_IN_TYPE> outData(expectedOutdataInfo.nelms());
  std::vector<DATA_IN_TYPE> expectedOutData(expectedOutdataInfo.nelms());
  for (int64_t d0 = 0; d0 < expectedOutShape[0]; ++d0) {
    for (int64_t d1 = 0; d1 < expectedOutShape[1]; ++d1) {
      for (int64_t d2 = 0; d2 < expectedOutShape[2]; ++d2) {
        for (int64_t d3 = 0; d3 < expectedOutShape[3]; ++d3) {
          for (int64_t d4 = 0; d4 < expectedOutShape[4]; ++d4) {

            // Derive the expected value
            int64_t value =
                d4 +
                inShape[3] *
                    (d3 + inShape[2] * (indices[d2 + indicesShape[1] * d1] +
                                        inShape[1] * d0));

            // Compute the array index
            int64_t index =
                d4 + expectedOutShape[4] *
                         (d3 + expectedOutShape[3] *
                                   (d2 + expectedOutShape[2] *
                                             (d1 + expectedOutShape[1] * d0)));

            expectedOutData[index] = static_cast<DATA_IN_TYPE>(value);
          }
        }
      }
    }
  }

  ConstVoidData constInData      = {inData.data(), inDataInfo};
  ConstVoidData constIndicesData = {indices.data(), inIndicesInfo};

  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto inId        = aiOnnx.constant(constInData, "constInData");
  auto indicesId   = aiOnnx.constant(constIndicesData, "constIndicesData");
  auto gatherOutId = aiOnnx.gather({inId, indicesId}, axis);

  // Dummy add op appended at the end since gather will not be folded if the
  // output is an anchor
  Shape add0Shape = {1};
  TensorInfo add0DataInfo(data_in_type, add0Shape);
  std::vector<DATA_IN_TYPE> add0Data(1);
  add0Data[0]                 = static_cast<DATA_IN_TYPE>(0);
  ConstVoidData constAdd0Data = {add0Data.data(), add0DataInfo};

  auto add0Id = aiOnnx.constant(constAdd0Data, "constAdd0Data");
  auto outId  = aiOnnx.add({gatherOutId, add0Id});

  builder->addOutputTensor(outId);

  popart::NDArrayWrapper<DATA_IN_TYPE> output(outData.data(), expectedOutShape);
  std::map<popart::TensorId, popart::IArray &> anchors = {{outId, output}};

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{outId, art}});
  auto device   = createTestDevice(TEST_TARGET);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      InputShapeInfo(),
      {}, // no SessionOptions
      Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

  session->prepareDevice();
  popart::StepIO stepio({}, anchors);
  session->run(stepio);

  // Check that the values are as expected
  BOOST_CHECK_EQUAL_COLLECTIONS(outData.begin(),
                                outData.end(),
                                expectedOutData.begin(),
                                expectedOutData.end());
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Gather_Types) {
  // ConstExprTest_Gather_Type<uint32_t>("UINT32");
  // ConstExprTest_Gather_Type<uint64_t>("UINT64");
  ConstExprTest_Gather_Type<int32_t, int32_t>("INT32", "INT32");
  ConstExprTest_Gather_Type<int32_t, int64_t>("INT32", "INT64");
  // ConstExprTest_Gather_Type<int64_t>("INT64");
  ConstExprTest_Gather_Type<popart::float16_t, int32_t>("FLOAT16", "INT32");
  ConstExprTest_Gather_Type<popart::float16_t, int64_t>("FLOAT16", "INT64");
  ConstExprTest_Gather_Type<float, int32_t>("FLOAT", "INT32");
  ConstExprTest_Gather_Type<float, int64_t>("FLOAT", "INT64");
  // ConstExprTest_Gather_Type<double>("DOUBLE");
}
