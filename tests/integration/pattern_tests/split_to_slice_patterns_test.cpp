// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SplitToSlicePatternTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

BOOST_AUTO_TEST_CASE(SplitToSliceTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{6}};

  auto input1 = builder->addInputTensor(shape1);
  auto ident0 = aiOnnx.identity({input1});

  auto outs = aiOnnx.split({ident0}, 3, 0, {1, 2, 3});

  for (auto out : outs) {
    builder->addOutputTensor(out);
  }

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1,
                           {{outs.at(0), AnchorReturnType("All")},
                            {outs.at(1), AnchorReturnType("All")},
                            {outs.at(2), AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Off;

  auto device = createTestDevice(TEST_TARGET, 2);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              userOptions,
              Patterns::create({"SplitOp"}).enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Slice_1).size() == 3);
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Split_2).size() == 0);
}
