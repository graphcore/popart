// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE AcosOpPatternTest

#include <functional>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(AcosOpPatternTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset7();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{2, 2}};
  float dummy1[4]      = {1.0, 2.0, 3.0, 4.0};
  ConstVoidData t1Data = {dummy1, shape1};
  auto input1          = builder->addInitializedInputTensor(t1Data);

  auto out = aiOnnx.acos({input1});

  builder->addOutputTensor(out);
  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

  SessionOptions userOptions;
  auto device = createTestDevice(TEST_TARGET, 2);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              userOptions,
              Patterns({PreAliasPatternType::AcosOpPattern})
                  .enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Acos_7).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Asin_7).size() == 1);
}

BOOST_AUTO_TEST_CASE(AcosOpPatternTest1) {

  using namespace popart;

  // check that the pattern is only on for level ALL and DEFAULT

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isAcosOpPatternEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isAcosOpPatternEnabled() == true);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isAcosOpPatternEnabled() == true);
}
