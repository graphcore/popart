// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DepthToSpaceOpPatternTest

#include <functional>

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(DepthToSpaceOpPatternTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset11();

  int N = 3;
  int C = 4;
  int H = 8;
  int W = 8;
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{N, C, H, W}};
  auto input = builder->addInputTensor(inputInfo);

  auto out = aiOnnx.depthtospace({input}, {2}, {"DCR"});

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
              Patterns({PreAliasPatternType::DepthToSpaceOpPattern})
                  .enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::DepthToSpace_11).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Reshape_5).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Transpose_1).size() == 1);
}

BOOST_AUTO_TEST_CASE(DepthToSpaceOpPatternTest1) {

  using namespace popart;

  // check that the pattern is only on for level ALL and DEFAULT

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isDepthToSpaceOpPatternEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isDepthToSpaceOpPatternEnabled() == true);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isDepthToSpaceOpPatternEnabled() == true);
}

BOOST_AUTO_TEST_CASE(SpaceToDepthOpPatternTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset11();

  int N = 3;
  int C = 4;
  int H = 8;
  int W = 8;
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{N, C, H, W}};
  auto input = builder->addInputTensor(inputInfo);

  auto out = aiOnnx.spacetodepth({input}, {2});

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
              Patterns({PreAliasPatternType::SpaceToDepthOpPattern})
                  .enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::SpaceToDepth_1).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Reshape_5).size() == 2);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Transpose_1).size() == 1);
}

BOOST_AUTO_TEST_CASE(SpaceToDepthOpPatternTest1) {

  using namespace popart;

  // check that the pattern is only on for level ALL and DEFAULT

  Patterns noPatterns(PatternsLevel::NoPatterns);
  noPatterns.enableRuntimeAsserts(false);
  BOOST_CHECK(noPatterns.isSpaceToDepthOpPatternEnabled() == false);

  Patterns defPatterns(PatternsLevel::Default);
  BOOST_CHECK(defPatterns.isSpaceToDepthOpPatternEnabled() == true);

  Patterns allPatterns(PatternsLevel::All);
  BOOST_CHECK(allPatterns.isSpaceToDepthOpPatternEnabled() == true);
}
