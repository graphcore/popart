// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Concat1InplaceTest

#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <filereader.hpp>
#include <memory>
#include <string>
#include <testdevice.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/sgd.hpp>
#include <popart/tensorinfo.hpp>

#include "popart/builder.gen.hpp"
#include "popart/names.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_concat1) {

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{1, 3}};
  auto in0 = builder->addInputTensor(shape0);
  auto s0  = aiOnnx.sigmoid({in0});
  auto c0  = aiOnnx.concat({s0, s0, s0}, 0);
  auto s1  = aiOnnx.relu({c0});
  auto out = aiOnnx.reducesum({s1}, std::vector<int64_t>{});

  out = builder->aiGraphcoreOpset1().l1loss({out}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              out,
              &optimizer,
              *device,
              {},
              Patterns(PatternsLevel::NoPatterns)
                  .enableRuntimeAsserts(false)
                  .enableInPlace(true)});

  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Relu).size() == 0);
}
