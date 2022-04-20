// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_irquery_tensorindextestwrapper

#include <boost/test/unit_test.hpp>
#include <string>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include "popart/graphid.hpp"
#include "popart/tensor.hpp"
#include "popart/tensors.hpp"
#include "testutil/irquery/irquery.hpp"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wkeyword-macro"
#endif

using namespace popart;
using namespace popart::irquery;

BOOST_AUTO_TEST_CASE(tensorindextestwrapper_index) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TensorIndexTestWrapper tw{ir, {5, t0}, "op", "input", "inputs"};

  BOOST_REQUIRE(5 == tw.index());
}

BOOST_AUTO_TEST_CASE(tensorindextestwrapper_id) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TensorIndexTestWrapper tw{ir, {5, t0}, "op", "input", "inputs"};

  BOOST_REQUIRE(t0->id == tw.id());
}

BOOST_AUTO_TEST_CASE(tensorindextestwrapper_tensor) {

  Ir ir;
  ir.getMainGraph().getTensors().addActGrad("t0");
  auto t0 = ir.getMainGraph().getTensors().get("t0");
  TensorIndexTestWrapper tw{ir, {5, t0}, "op", "input", "inputs"};

  BOOST_REQUIRE(t0 == tw.tensor().unwrap());
}
