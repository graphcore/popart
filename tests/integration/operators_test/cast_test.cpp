// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CastTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/ir.hpp>
#include <popart/optimizer.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Cast_OutType_Equals_InType) {
  auto builder = Builder::create();
  auto t0      = builder->addInputTensor("INT32", std::vector<int64_t>{2, 2});
  auto t1      = builder->aiOnnxOpset9().cast({t0}, "INT32");

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              DataFlow(1, std::vector<std::string>{t1}),
              {},
              nullptr,
              *device,
              {},
              Patterns::create({"OpToIdentity"}).enableRuntimeAsserts(false)});

  // Check that the CastOp has been converted to an IdentityOp
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Cast).size() == 0);
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Identity).size() == 1);
}