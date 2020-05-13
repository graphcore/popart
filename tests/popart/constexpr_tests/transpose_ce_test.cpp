// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprTransposeTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

// more constexpr transpose tests in fp16_test.py
BOOST_AUTO_TEST_CASE(ConstExprTest_Transpose1) {

  Shape inShape = {2, 5};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {5, 2};
  std::vector<int> rawConstInputData(5 * 2);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  popart::NDArrayWrapper<int> constData(rawConstInputData.data(), {5, 2});

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constShape}};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto constId = aiOnnx.constant(constShapeData, "out0ShapeData");
  auto inId    = builder->addInputTensor(inInfo);

  auto outShapeId = aiOnnx.transpose({constId}, {});
  auto out        = aiOnnx.add({outShapeId, inId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{out, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      {},
      popart::Patterns({popart::PreAliasPatternType::PostNRepl}));

  // prepare the anchors
  int rawOutputData[10];
  popart::NDArrayWrapper<int> outData(rawOutputData, {2, 5});

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  int rawInputData[10] = {
      0,
  };
  popart::NDArrayWrapper<int> inData(rawInputData, {2, 5});
  std::map<popart::TensorId, popart::IArray &> inputs = {{inId, inData}};

  popart::StepIO stepio(inputs, anchors);

  session->run(stepio);

  popart::logging::ir::err("const : {}", constData);
  popart::logging::ir::err("input : {}", inData);
  popart::logging::ir::err("output : {}", outData);

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Transpose2) {

  Shape inShape = {4, 2, 3};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {2, 3, 4};
  std::vector<int> rawConstInputData(4 * 2 * 3);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  popart::NDArrayWrapper<int> constData(rawConstInputData.data(), {2, 3, 4});

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constShape}};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto constId = aiOnnx.constant(constShapeData, "constShapeData");
  auto inId    = builder->addInputTensor(inInfo);

  auto outShapeId = aiOnnx.transpose({constId}, {2, 0, 1});
  auto out        = aiOnnx.add({outShapeId, inId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{out, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      popart::InputShapeInfo(),
      {},
      popart::Patterns({popart::PreAliasPatternType::PostNRepl}));

  // prepare the anchors
  int rawOutputData[24];
  popart::NDArrayWrapper<int> outData(rawOutputData, {4, 2, 3});

  std::map<popart::TensorId, popart::IArray &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  // prepare the inputs
  int rawInputData[24] = {
      0,
  };
  popart::NDArrayWrapper<int> inData(rawInputData, {4, 2, 3});
  std::map<popart::TensorId, popart::IArray &> inputs = {{inId, inData}};

  popart::StepIO stepio(inputs, anchors);

  session->run(stepio);

  popart::logging::ir::err("const : {}", constData);
  popart::logging::ir::err("input : {}", inData);
  popart::logging::ir::err("output : {}", outData);

  int expectedOutput[24] = {1, 5, 9,  13, 17, 21, 2, 6, 10, 14, 18, 22,
                            3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24};

  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[24],
                         &rawOutputData[0]) == true);
}
