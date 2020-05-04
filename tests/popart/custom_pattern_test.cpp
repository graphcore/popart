// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CustomPatternTest

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/graph.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/relu.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/session.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

class ReplaceReluWithNeg : public PreAliasPattern {
public:
  bool matches(Op *op) const override { return op->isConvertibleTo<ReluOp>(); }

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override {
    logging::debug("ReplaceReluWithNeg::apply({})", op->debugName());
    op->setName("someReluOp");

    auto negOp =
        makeReplacementOpInIr(Onnx::Operators::Neg_6, op, "reluReplacement");

    auto inputId  = op->inId(ReluOp::getInIndex());
    auto outputId = op->outId(ReluOp::getOutIndex());
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    op->getGraph().eraseOp(op->id);

    negOp->connectInTensor(NegateOp::getInIndex(), inputId);
    negOp->connectOutTensor(NegateOp::getOutIndex(), outputId);
    negOp->setup();

    return true;
  }
};

namespace {
static PatternCreator<ReplaceReluWithNeg> myPatternCreator("ReplaceReluWithNeg",
                                                           true);
}

// Use a custom pattern to replace a ReluOp with a NegateOp.
BOOST_AUTO_TEST_CASE(CustomPattern) {
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

  auto session =
      popart::InferenceSession::createFromOnnxModel(proto, dataFlow, device);

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

  // This should only be correct if the ReluOp was replaced with a NegateOp.
  BOOST_CHECK(rawOutputData[0] == -2.0f);
  BOOST_CHECK(rawOutputData[1] == -4.0f);
}
