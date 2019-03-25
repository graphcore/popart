#define BOOST_TEST_MODULE ConstExprTransposeTest

#include <boost/test/unit_test.hpp>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

// more constexpr transpose tests in fp16_test.py
BOOST_AUTO_TEST_CASE(ConstExprTest_Transpose1) {

  Shape inShape = {2, 5};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {5, 2};
  std::vector<int> rawConstInputData(5 * 2);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  poponnx::NDArrayWrapper<int> constData(rawConstInputData.data(), {5, 2});

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
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{out, art}});

  auto session = poponnx::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      {},
      poponnx::InputShapeInfo(),
      {},
      poponnx::Patterns({poponnx::PreAliasPatternType::POSTNREPL}));

  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();
  session->setDevice(cpuDevice);

  // prepare the anchors
  int rawOutputData[10];
  poponnx::NDArrayWrapper<int> outData(rawOutputData, {2, 5});

  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  int rawInputData[10] = {
      0,
  };
  poponnx::NDArrayWrapper<int> inData(rawInputData, {2, 5});
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{inId, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->run(stepio);

  poponnx::logging::ir::err("const : {}", constData);
  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);

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

  poponnx::NDArrayWrapper<int> constData(rawConstInputData.data(), {2, 3, 4});

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
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{out, art}});

  auto session = poponnx::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      {},
      poponnx::InputShapeInfo(),
      {},
      poponnx::Patterns({poponnx::PreAliasPatternType::POSTNREPL}));

  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();
  session->setDevice(cpuDevice);

  // prepare the anchors
  int rawOutputData[24];
  poponnx::NDArrayWrapper<int> outData(rawOutputData, {4, 2, 3});

  std::map<poponnx::TensorId, poponnx::IArray &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  // prepare the inputs
  int rawInputData[24] = {
      0,
  };
  poponnx::NDArrayWrapper<int> inData(rawInputData, {4, 2, 3});
  std::map<poponnx::TensorId, poponnx::IArray &> inputs = {{inId, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->run(stepio);

  poponnx::logging::ir::err("const : {}", constData);
  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);

  int expectedOutput[24] = {1, 5, 9,  13, 17, 21, 2, 6, 10, 14, 18, 22,
                            3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24};

  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[24],
                         &rawOutputData[0]) == true);
}
