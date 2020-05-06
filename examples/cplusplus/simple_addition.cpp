// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

//
// A simple addition that returns its result.
// Similar to the Python simple_addition example
//

// Compile with
// g++ -std=c++11 simple_addition.cpp -lpopart -DONNX_NAMESPACE=onnx -o
// simple_addition

#include <iostream>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/logging.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op.hpp>
#include <popart/op/l1.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/session.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

auto main(int argc, char **argv) -> int {

  // Generate an ONNX inference model
  auto builder = popart::Builder::create();

  std::cout << "Setting Opset9\n";
  auto aiOnnx = builder->aiOnnxOpset9();

  // Add input tensors
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};
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
  auto dataFlow = popart::DataFlow(1, {{o, popart::AnchorReturnType("ALL")}});

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "1"}};
  auto ipuModelDevice =
      popart::DeviceManager::createDeviceManager().createIpuModelDevice(
          deviceOpts);
  // or acquireAvailableDevice();

  std::cout << "Creating session from Onnx Model...\n";
  auto session = popart::InferenceSession::createFromOnnxModel(
      proto, dataFlow, ipuModelDevice);
  std::cout << "Creating session from Onnx Model...done\n";

  // Prepare input tensor
  float rawInputData[2] = {2.0f, 4.0f};
  popart::NDArrayWrapper<float> inDataA(rawInputData, {2});
  popart::NDArrayWrapper<float> inDataB(rawInputData, {2});
  std::map<popart::TensorId, popart::IArray &> inputs = {{a, inDataA},
                                                         {b, inDataB}};

  // Prepare output tensor
  float rawOutputData[2] = {0, 0};
  popart::NDArrayWrapper<float> outData(rawOutputData, {2});
  std::map<popart::TensorId, popart::IArray &> anchors = {{o, outData}};

  std::cout << "Preparing session device...\n";
  session->prepareDevice();
  std::cout << "Preparing session device...done\n";

  popart::StepIO stepio(inputs, anchors);

  std::cout << "Running..."
            << "\n";
  session->run(stepio);
  std::cout << "Running...done"
            << "\n";

  std::cout << "Input Data:  " << inDataA << "\n";
  std::cout << "Input Data:  " << inDataB << "\n";
  std::cout << "Output Data: " << outData << "\n";

  popart::logging::ir::err("inputs : {}, {}", inDataA, inDataB);
  popart::logging::ir::err("output : {}", outData);
}
