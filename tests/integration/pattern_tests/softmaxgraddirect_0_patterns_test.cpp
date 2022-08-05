// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SoftmaxGradDirectTest0

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <set>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

#include "popart/builder.gen.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(SoftmaxGradDirect0) {
  // (label), (probs) -> [NLLGrad]
  // [NllGrad] -> (d_probs)
  // (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts)
  //
  // should become
  // (label), (probs) -> [SoftmaxGradDirect] -> (d_acts)
  //
  // but ONLY if "they're" on the same IPUs
  //
  // If NlllWithSoftmaxGradDirect pattern is enabled,
  // and since the loss is anchored, this should become
  //
  // (label), (probs) -> [NlllWithSoftmaxGradDirect] -> (loss), (d_acts)

  auto test = [](bool sameIPU, bool enableNlllWithSoftmaxGradDirect) {
    // Build an onnx model
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();

    TensorInfo inInfo{"FLOAT", std::vector<int64_t>{2, 2}};
    TensorInfo labelInfo{"INT32", std::vector<int64_t>{2}};

    auto input1 = builder->addInputTensor(inInfo);
    auto input2 = builder->addInputTensor(labelInfo);

    auto identOut   = aiOnnx.identity({input1});
    auto softmaxOut = aiOnnx.softmax({identOut});
    auto nlll = aiGraphcore.nllloss({softmaxOut, input2}, ReductionType::Sum);

    if (sameIPU == false) {
      builder->virtualGraph(identOut, 2);
      builder->virtualGraph(softmaxOut, 2);
      builder->virtualGraph(nlll, 1);
    }

    builder->addOutputTensor(nlll);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    // Add the last tensor, and the 3rd tensor as anchors
    auto art = AnchorReturnType("All");
    auto dataFlow =
        DataFlow(1, {{reservedGradientPrefix() + input1, art}, {nlll, art}});
    auto optimizer = ConstSGD(0.01);

    auto opts = SessionOptions();
    if (sameIPU == false) {
      opts.virtualGraphMode = VirtualGraphMode::Manual;
    }

    // No .dot files will be written
    opts.dotChecks = {};

    auto device = createTestDevice(TEST_TARGET);

    auto patterns = Patterns::create({"PreUniRepl", "SoftmaxGradDirect"})
                        .enableRuntimeAsserts(false);
    patterns.enableNlllWithSoftMaxGradDirect(enableNlllWithSoftmaxGradDirect);

    Ir ir;
    ir.prepare(
        {modelProto, {}, dataFlow, nlll, &optimizer, *device, opts, patterns});

    // Check the ir

    // NllGradOp and SoftmaxGradOp should have been replaced with
    // SoftmaxGradDirectOp, but ONLY if same IPUs
    if (sameIPU == true && enableNlllWithSoftmaxGradDirect == false) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    } else if (sameIPU == false && enableNlllWithSoftmaxGradDirect == false) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    }

    // NllOp, NllGradOp and SoftmaxGradOp should have been replaced with
    // NlllWithSoftmaxGradDirectOp, but ONLY if same IPUs
    if (sameIPU == true && enableNlllWithSoftmaxGradDirect == true) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 0);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 1);
    } else if (sameIPU == false && enableNlllWithSoftmaxGradDirect == true) {
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::Nll).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomGradOperators::NllGrad).size() == 1);
      BOOST_CHECK(ir.opsOfType(Onnx::GradOperators::SoftmaxGrad).size() == 1);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::SoftmaxGradDirect).size() ==
          0);
      BOOST_CHECK(
          ir.opsOfType(Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect)
              .size() == 0);
    }
  };

  test(true, false);
  test(false, false);

  test(true, true);
  test(false, true);
}
