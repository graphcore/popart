// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SplitGatherTest0

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

BOOST_AUTO_TEST_CASE(SplitGatherTest0) {

  using namespace popart;

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape1{"FLOAT", std::vector<int64_t>{2, 2}};
  TensorInfo shape2{"INT32", std::vector<int64_t>{2}};

  float dummy[4]       = {0, 0, 0, 0};
  ConstVoidData t1Data = {dummy, shape1};
  auto input1          = builder->addInitializedInputTensor(t1Data);
  auto input2          = builder->addInputTensor(shape2);

  auto out = aiOnnx.gather({input1, input2}, 1);

  builder->virtualGraph(out, 0);
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});

  SessionOptions userOptions;
  userOptions.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET, 2);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},
              nullptr,
              *device,
              userOptions,
              Patterns::create({"SplitGather"}).enableRuntimeAsserts(false)});

  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Gather_1).size() == 2);
}
