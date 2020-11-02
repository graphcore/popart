// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AtanhOpPatternTest

#include <functional>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(AtanhOpPatternTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2, 2}};
  auto input = builder->addInputTensor(inputInfo);

  auto out = aiOnnx.atanh({input});

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
              Patterns({PreAliasPatternType::AtanhOpPattern})
                  .enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Atanh_9).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Log_6).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Mul_7).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Div_7).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Add_7).size() == 1);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Sub_7).size() == 1);
}

BOOST_AUTO_TEST_CASE(AtanhOpPatternTest1) {

  using namespace popart;

  // check that the pattern is only on for level ALL and DEFAULT

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isAtanhOpPatternEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isAtanhOpPatternEnabled() == true);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isAtanhOpPatternEnabled() == true);
}
