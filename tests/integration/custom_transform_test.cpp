// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CustomTransformTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/add.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/relu.hpp>
#include <popart/session.hpp>

#include "popart/builder.gen.hpp"
#include "popart/dataflow.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/stepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Graph;
} // namespace popart

using namespace popart;
class DoubleInput : public Transform {
public:
  static std::size_t id() { return typeid(DoubleInput).hash_code(); }

  DoubleInput() : Transform() {}
  virtual ~DoubleInput() override {}

  virtual bool apply(Graph &graph) const final {
    auto addOp = graph.createOp<AddOp>(Onnx::Operators::Add_7,
                                       Op::Settings(graph, "add"));
    addOp->connectInTensor(AddOp::getArg0InIndex(), "input");
    addOp->connectInTensor(AddOp::getArg1InIndex(), "input");
    addOp->createAndConnectOutTensor(AddOp::getOutIndex(), "addOut");
    addOp->setup();

    Tensor *input = graph.getTensors().get("input");
    auto reluOp   = input->consumers.getOps()[0];
    reluOp->disconnectAllInputs();
    reluOp->connectInTensor(ReluOp::getInIndex(), "addOut");
    return true;
  }

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "DoubleInput"; }
};

namespace {
bool init = Transform::registerTransform(new DoubleInput);
}

// Use a custom transorform to Print ops debug info
BOOST_AUTO_TEST_CASE(customTransform) {
  // Generate a model that is a single relu op.
  auto builder = popart::Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};
  auto input = builder->addInputTensor(inputInfo);

  auto reluOut = aiOnnx.relu({input});

  builder->addOutputTensor(reluOut);

  auto proto = builder->getModelProto();

  // Create the session and run a session.
  auto dataFlow =
      popart::DataFlow(1, {{reluOut, popart::AnchorReturnType("All")}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto opts = SessionOptions();
  opts.experimentalSettings.customTransformApplierSettings.insert(
      {"Fwd0", {"DoubleInput", "Prune"}});

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto, dataFlow, device, {}, opts);

  float rawOutputData[2] = {0, 0};
  popart::NDArrayWrapper<float> outData(rawOutputData, {2});

  std::map<popart::TensorId, popart::IArray &> anchors = {{reluOut, outData}};

  session->prepareDevice();

  // prepare the input tensor for this example
  float rawInputData[2] = {2.0f, 4.0f};
  popart::NDArrayWrapper<float> inData(rawInputData, {2});
  std::map<popart::TensorId, popart::IArray &> inputs = {{input, inData}};

  popart::StepIO stepio(inputs, anchors);
  session->run(stepio);

  // This should only be correct if the input is doubled.
  BOOST_CHECK(rawOutputData[0] == 4.0f);
  BOOST_CHECK(rawOutputData[1] == 8.0f);
}
