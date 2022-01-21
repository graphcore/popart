// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Willow_Op_Call
#include <boost/test/unit_test.hpp>

#include <popart/ir.hpp>
#include <popart/op/casttofp8.hpp>
#include <popart/op/init.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestCastToFp8Op) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  int nBitMantissa = 4;
  int nBitExponent = 3;
  int exponentBias = 7;

  TensorId in = "fp16-tensor";
  g.getTensors().addStream(in, {DataType::FLOAT16, Shape{100}});
  BOOST_REQUIRE_NO_THROW(g.createConnectedOp<CastToFp8Op>(
      {{CastToFp8Op::getInIndex(), in}},
      {{CastToFp8Op::getOutIndex(), "fp8-tensor"}},
      Onnx::CustomOperators::CastToFp8,
      nBitMantissa,
      nBitExponent,
      exponentBias,
      Op::Settings{g, "CastToFp8"}));
}
