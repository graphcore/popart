// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Willow_Op_Call
#include <boost/test/unit_test.hpp>

#include <popart/ir.hpp>
#include <popart/op/castfromfp8.hpp>
#include <popart/op/init.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestCastFromFp8Op) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  int nBitMantissa = 4;
  int nBitExponent = 3;
  int exponentBias = 7;
  DataType type    = DataType::FLOAT16;

  TensorId in = "fp8-tensor";
  g.getTensors().addStream(in, {DataType::FLOAT8, Shape{100}});
  BOOST_REQUIRE_NO_THROW(g.createConnectedOp<CastFromFp8Op>(
      {{CastFromFp8Op::getInIndex(), in}},
      {{CastFromFp8Op::getOutIndex(), "fp16-tensor"}},
      Onnx::CustomOperators::CastFromFp8,
      type,
      nBitMantissa,
      nBitExponent,
      exponentBias,
      Op::Settings{g, "CastFromFp8"}));
}