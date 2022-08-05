// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE StepIOTest

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <boost/type_index/type_index_facade.hpp>
#include <chrono>
#include <cstdint>
#include <filereader.hpp>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <testdevice.hpp>
#include <thread>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>

#include "popart/builder.gen.hpp"
#include "popart/datatype.hpp"
#include "popart/logging.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

namespace {

std::shared_ptr<popart::DeviceInfo> acquireIpu() {
  // keep trying to attach to a device until one is available (this may not
  // always be the case as other tests might be running in parallel).
  while (true) {
    if (auto d = createTestDevice(TEST_TARGET, 1, 1216)) {
      return d;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return nullptr;
}

} // unnamed namespace

BOOST_AUTO_TEST_CASE(StepIOTest_BufferInput) {

  Shape inShape = {2, 5};
  TensorInfo inInfo{"INT32", inShape};

  std::vector<int> rawConstInputData(5 * 2);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  popart::NDArrayWrapper<int> constData(rawConstInputData.data(), {5, 2});

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constData.shape()}};

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
      popart::Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

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

  popart::logging::info("const : {}", constData);
  popart::logging::info("input : {}", inData);
  popart::logging::info("output : {}", outData);

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}

bool ipu_available(boost::unit_test::test_unit_id) {
  auto devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices();
  return devices.size() > 0;
}

BOOST_AUTO_TEST_CASE(StepIOTest_BufferInput_Ipu,
                     *boost::unit_test::precondition(ipu_available)) {

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

  auto ipuDevice = acquireIpu();

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      ipuDevice,
      popart::InputShapeInfo(),
      {},
      popart::Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

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

  popart::logging::info("const : {}", constData);
  popart::logging::info("input : {}", inData);
  popart::logging::info("output : {}", outData);

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}

BOOST_AUTO_TEST_CASE(StepIOTest_CallbackInput) {

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
      popart::Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

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

  popart::StepIOCallback::InputCallback input_callback =
      [&](TensorId id, bool prefetch) -> ConstVoidData {
    popart::logging::info("input callback called {}", id);
    (void)prefetch;
    popart::NDArrayWrapper<int> inData(rawInputData, {2, 5});

    ConstVoidData data;
    data.data = inData.data();
    data.info = TensorInfo(DataType::INT32, {2, 5});
    return data;
  };

  popart::StepIOCallback::InputCompleteCallback input_complete_callback =
      [](TensorId id) -> void {
    popart::logging::info("input complete callback called {}", id);
  };

  popart::StepIOCallback::OutputCallback output_callback =
      [&](TensorId id) -> MutableVoidData {
    popart::logging::info("output callback called {}", id);

    popart::NDArrayWrapper<int> outData(rawOutputData, {2, 5});

    MutableVoidData data;
    data.data = outData.data();
    data.info = TensorInfo(DataType::INT32, {2, 5});
    return data;
  };

  popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
      [](TensorId id) -> void {
    popart::logging::info("output complete callback called {}", id);
  };

  popart::StepIOCallback stepio(input_callback,
                                input_complete_callback,
                                output_callback,
                                output_complete_callback);

  session->run(stepio);

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}

BOOST_AUTO_TEST_CASE(StepIOTest_CallbackInput_Ipu,
                     *boost::unit_test::precondition(ipu_available)) {

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

  auto ipuDevice = acquireIpu();

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      ipuDevice,
      popart::InputShapeInfo(),
      {},
      popart::Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

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

  popart::StepIOCallback::InputCallback input_callback =
      [&](TensorId id, bool prefetch) -> ConstVoidData {
    popart::logging::info("input callback called {}", id);
    (void)prefetch;
    popart::NDArrayWrapper<int> inData(rawInputData, {2, 5});

    ConstVoidData data;
    data.data = inData.data();
    data.info = TensorInfo(DataType::INT32, {2, 5});
    return data;
  };

  popart::StepIOCallback::InputCompleteCallback input_complete_callback =
      [](TensorId id) -> void {
    popart::logging::info("input complete callback called {}", id);
  };

  popart::StepIOCallback::OutputCallback output_callback =
      [&](TensorId id) -> MutableVoidData {
    popart::logging::info("output callback called {}", id);

    popart::NDArrayWrapper<int> outData(rawOutputData, {2, 5});

    MutableVoidData data;
    data.data = outData.data();
    data.info = TensorInfo(DataType::INT32, {2, 5});
    return data;
  };

  popart::StepIOCallback::OutputCompleteCallback output_complete_callback =
      [](TensorId id) -> void {
    popart::logging::info("output complete callback called {}", id);
  };

  popart::StepIOCallback stepio(input_callback,
                                input_complete_callback,
                                output_callback,
                                output_complete_callback);

  session->run(stepio);

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}
