// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprReduceProdTest

#include <algorithm>
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
#include <popart/names.hpp>
#include <popart/testdevice.hpp>

#include "popart/builder.gen.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"
#include "popart/voiddata.hpp"

using namespace popart;

const Shape SHAPE{2, 3, 5};
const int64_t RANK = SHAPE.size();

void check_combination(nonstd::optional_lite::optional<Shape> axes,
                       int64_t keepdims) {
  // Print out axes and keepdims
  BOOST_TEST_MESSAGE("keepdims: " + std::to_string(keepdims));
  std::string axes_string;
  if (axes.has_value()) {
    for (auto axis : axes.value()) {
      axes_string += std::to_string(axis) + " ";
    }
    BOOST_TEST_MESSAGE("axes: " + axes_string);
  } else
    BOOST_TEST_MESSAGE("no axes provided");
  BOOST_TEST_MESSAGE("");

  // Build the graph
  std::vector<float> raw_const_data(SHAPE[0] * SHAPE[1] * SHAPE[2], 1.0);
  ConstVoidData const_data = {raw_const_data.data(), {"FLOAT", SHAPE}};

  TensorInfo X_info{"FLOAT", SHAPE};

  auto builder = Builder::create();
  auto X       = builder->addInputTensor(X_info);
  auto aiOnnx  = builder->aiOnnxOpset11();
  auto C       = aiOnnx.constant(const_data);

  TensorId reduceprod_node_1, reduceprod_node_2;
  reduceprod_node_1 = aiOnnx.reduceprod({C}, axes, keepdims);
  reduceprod_node_2 = aiOnnx.reduceprod({X}, axes, keepdims);

  auto output = aiOnnx.add({reduceprod_node_1, reduceprod_node_2});

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  auto data_flow = DataFlow(1, {{output, AnchorReturnType("All")}});
  auto device    = createTestDevice(TEST_TARGET);

  // Create the IR, adding output as an anchor
  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              TensorId(),
              nullptr,
              *device,
              {}, // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  // 1) check that the Add Op is present
  BOOST_CHECK_MESSAGE(
      ir.opsOfType(Onnx::AiOnnx::OpSet11::Add).size() == 1,
      "Number of Add nodes is incorrect: " +
          std::to_string(ir.opsOfType(Onnx::AiOnnx::OpSet11::Add).size()));
  // 2) check that one of the ReduceProd ops has been constfolded
  BOOST_CHECK_MESSAGE(
      ir.opsOfType(Onnx::AiOnnx::OpSet11::ReduceProd).size() == 1,
      "Number of ReduceProd nodes is incorrect: " +
          std::to_string(
              ir.opsOfType(Onnx::AiOnnx::OpSet11::ReduceProd).size()));
  // This test should also error out if constant folding produces a different
  // shape than ReduceProd on X
}

BOOST_AUTO_TEST_CASE(ConstExprTest_ReduceProd) {
  // clang-format off
  //
  // With C being a constant, this test contructs the following graph:
  //
  // (C) --> [ReduceProd] --> [Add] --> (*)
  // (X) --> [ReduceProd] ----^
  //
  // After constant folding the graph should look like this:
  //
  //          (reduced C) --> [Add] --> (*)
  // (X) --> [ReduceProd] ----^
  //
  // The reduction is tried over all possible combinations of axes and keepdims values for ReduceProd.
  //
  // clang-format on

  // positive axes
  for (int combination = 0; combination < (1 << RANK); ++combination) {
    Shape axes;
    for (int i = 0; i < RANK; ++i) {
      if ((combination >> i) & 1)
        axes.push_back(i);
    }
    check_combination(axes, 0);
    check_combination(axes, 1);
  }

  // negative axes
  for (int combination = 1; combination < (1 << RANK); ++combination) {
    Shape axes;
    for (int i = 0; i < RANK; i++) {
      if ((combination >> i) & 1)
        axes.push_back(-i - 1);
    }
    check_combination(axes, 0);
    check_combination(axes, 1);
  }

  // no axes
  check_combination(nonstd::nullopt, 0);
  check_combination(nonstd::nullopt, 1);

  // duplicated axes
  check_combination(Shape{1, 1}, 0);
  check_combination(Shape{1, -2}, 1);

  // mixed negative and positive axes
  check_combination(Shape{1, -1}, 0);
  check_combination(Shape{1, -3}, 1);
}
