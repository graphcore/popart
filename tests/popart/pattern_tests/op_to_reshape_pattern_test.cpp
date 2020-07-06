// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OpToReshapePatternTest

#include <functional>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(OpToReshapeTest0) {

  using namespace popart;

  typedef std::function<TensorId(AiOnnxOpset9 &, TensorId &)> OpCall;

  auto opsToTest = {OpCall([](AiOnnxOpset9 &opset, TensorId &input1) {
                      return opset.squeeze({input1}, {1});
                    }),
                    OpCall([](AiOnnxOpset9 &opset, TensorId &input1) {
                      return opset.unsqueeze({input1}, {1});
                    }),
                    OpCall([](AiOnnxOpset9 &opset, TensorId &input1) {
                      return opset.flatten({input1});
                    })};

  for (auto opUnderTest : opsToTest) {

    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo shape1{"FLOAT", std::vector<int64_t>{6, 1, 3, 1}};

    auto input1 = builder->addInputTensor(shape1);

    auto out = opUnderTest(aiOnnx, input1);
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

    SessionOptions userOptions;

    auto device = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {},
                nullptr,
                *device,
                userOptions,
                Patterns({PreAliasPatternType::OpToReshape})
                    .enableRuntimeAsserts(false)});

    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Squeeze_1).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Squeeze_11).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Unsqueeze_1).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Unsqueeze_11).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Flatten_1).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Flatten_9).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Flatten_11).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::Operators::Reshape_5).size() == 1);
  }
}

BOOST_AUTO_TEST_CASE(OpToReshapeTest1) {

  using namespace popart;

  // check that the pattern is only on for level ALL and DEFAULT

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isOpToReshapeEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isOpToReshapeEnabled() == true);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isOpToReshapeEnabled() == true);
}