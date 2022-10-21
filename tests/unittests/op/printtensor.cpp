// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PrintTensorOpTest

#include <boost/test/unit_test.hpp>
#include <popart/graph.hpp>
#include <popart/op/printtensor.hpp>
#include <popart/sessionoptions.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/ir.hpp"
#include "popart/op.hpp"

using namespace popart;

// Test that hasSideEffect is conditional on printSelf
BOOST_AUTO_TEST_CASE(PrintTensorOp_hasSideEffects_test) {
  SessionOptions opts;
  Ir ir;
  ir.setUserOptions(opts);
  Graph &g = ir.getMainGraph();

  Op::Settings opSettings(g, "");

  // printSelf == true
  auto op0 = g.createOp<PrintTensorOp>(
      Onnx::CustomOperators::PrintTensor_1, true, false, "", 1, opSettings);

  BOOST_TEST(op0->hasSideEffect());

  // printSelf == false
  auto op1 = g.createOp<PrintTensorOp>(
      Onnx::CustomOperators::PrintTensor_1, false, false, "", 1, opSettings);

  BOOST_TEST(!op1->hasSideEffect());
}
