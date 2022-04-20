// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

//
// A simple addition that returns its result.
// Similar to the Python simple_addition example
// Recreated as a test to demonstrate a simple example.

#define BOOST_TEST_MODULE simple_addition_test

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/dataflow.hpp"
#include "popart/names.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"

namespace popart {
class IArray;
} // namespace popart

using namespace popart;

int runTest(float rawInputData[2]) {

  // Generate an ONNX inference model
  auto builder = Builder::create();

  std::cout << "Setting Opset9\n";
  auto aiOnnx = builder->aiOnnxOpset9();

  // Add input tensors
  TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};
  std::cout << "Adding input tensor a\n";
  auto a = builder->addInputTensor(inputInfo);
  std::cout << "Adding input tensor b\n";
  auto b = builder->addInputTensor(inputInfo);

  // Add operation
  std::cout << "Adding operation add(a,b)\n";
  auto o = aiOnnx.add({a, b});

  // Add output tensor
  std::cout << "Adding output tensor o\n";
  builder->addOutputTensor(o);

  std::cout << "Getting model proto\n";
  auto proto = builder->getModelProto();

  std::cout << "Constructing DataFlow\n";
  auto dataFlow = DataFlow(1, {{o, AnchorReturnType("ALL")}});

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);
  // or acquireAvailableDevice();

  std::cout << "Creating session from Onnx Model...\n";
  auto session =
      InferenceSession::createFromOnnxModel(proto, dataFlow, ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";

  NDArrayWrapper<float> inDataA(rawInputData, {2});
  NDArrayWrapper<float> inDataB(rawInputData, {2});
  std::map<TensorId, IArray &> inputs = {{a, inDataA}, {b, inDataB}};

  // Prepare output tensor
  float rawOutputData[2] = {0, 0};
  NDArrayWrapper<float> outData(rawOutputData, {2});
  std::map<TensorId, IArray &> anchors = {{o, outData}};

  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";

  StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";

  std::cout << "Input Data:  " << inDataA << "\n";
  std::cout << "Input Data:  " << inDataB << "\n";
  std::cout << "Output Data: " << outData << "\n";

  logging::ir::info("inputs : {}, {}", inDataA, inDataB);
  logging::ir::info("output : {}", outData);

  BOOST_CHECK(outData[0] == 4.0f);
  BOOST_CHECK(outData[1] == 8.0f);

  return 0;
}

BOOST_AUTO_TEST_CASE(simple_addition_test) {
  float rawInputData[2] = {2.0f, 4.0f};
  runTest(rawInputData);
}
