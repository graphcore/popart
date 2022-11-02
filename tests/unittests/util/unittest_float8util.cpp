// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE unittest_float8util

#include "boost/test/auto_unit_test.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/ir.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorindex.hpp"
#include "popart/util/float8util.hpp"
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplar/Type.hpp>
#include <poputil/TileMapping.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(TestPopArtFloat8ToPoplarQuarter) {
  BOOST_CHECK(toPoplarQuarterFormat(DataType::FLOAT8_143) ==
              poplar::QuarterMetadata::Format::F143);
  BOOST_CHECK(toPoplarQuarterFormat(DataType::FLOAT8_152) ==
              poplar::QuarterMetadata::Format::F152);
}

BOOST_AUTO_TEST_CASE(TestReintepretCastUInt8ToQuarter) {
  poplar::IPUModel ipuModel;
  poplar::Device device = ipuModel.createDevice();
  poplar::Target target = device.getTarget();

  poplar::Graph graph(target);
  poplar::program::Sequence prog;

  auto in       = graph.addVariable(poplar::UNSIGNED_CHAR, {2, 2});
  int log2Scale = 10;
  graph.setTileMapping(in, 0);

  auto format = poplar::QuarterMetadata::Format::F143;
  auto out = reinterpretCastUInt8ToQuarter(graph, in, format, log2Scale, prog);

  BOOST_CHECK(out.elementType() == poplar::QUARTER);
  BOOST_CHECK(out.shape() == in.shape());

  // Should raise if the input is not type UNSIGNED_CHAR
  auto badIn = graph.addVariable(poplar::INT, {2, 3, 4});

  BOOST_CHECK_THROW(
      reinterpretCastUInt8ToQuarter(graph, badIn, format, log2Scale, prog),
      error);
}

BOOST_AUTO_TEST_CASE(TestCreateLog2ScaleInRangeProgram) {
  poplar::IPUModel ipuModel;
  poplar::Device device = ipuModel.createDevice();
  poplar::Target target = device.getTarget();

  poplar::Graph graph(target);

  int lower = -10;
  int upper = 10;

  auto inRange      = graph.addConstant(poplar::INT, {}, 0);
  auto outsideRange = graph.addConstant(poplar::INT, {}, 100);

  graph.setTileMapping(inRange, 0);
  graph.setTileMapping(outsideRange, 0);

  auto okProg = createAssertLog2ScaleInRangeProg(graph, inRange, lower, upper);
  auto throwProg =
      createAssertLog2ScaleInRangeProg(graph, outsideRange, lower, upper);

  BOOST_CHECK(!okProg.isEmpty());
  BOOST_CHECK(!throwProg.isEmpty());

  poplar::Engine engine(graph, {okProg, throwProg});
  engine.load(device);

  engine.run(0); // should run fine
  BOOST_CHECK_THROW(engine.run(1), poplar::application_runtime_error);
}

BOOST_AUTO_TEST_CASE(TestThrowsOnInvalidFloat8Inputs) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  std::string name = "testOp";
  TensorIndexMap inputs;

  std::string expectedMsg;
  auto errorMessageMatches = [&expectedMsg](popart::error const &error) {
    return std::string(error.what()).find(expectedMsg) != std::string::npos;
  };

  auto runTest =
      [&inputs, &name, &expectedMsg, &errorMessageMatches](
          Tensor &a, Tensor &b, Tensor &l2s, std::string shouldMatch) {
        inputs.insert(0, &a);
        inputs.insert(1, &b);
        inputs.insert(2, &l2s);

        expectedMsg = shouldMatch;
        BOOST_CHECK_EXCEPTION(validateOpFloat8Inputs(&inputs, 2, name),
                              error,
                              errorMessageMatches);
        inputs.clear();
      };

  Tensor a("a", TensorType::ActGrad, g);
  Tensor b("b", TensorType::ActGrad, g);
  Tensor log2Scale("log2scale", TensorType::ActGrad, g);
  a.info.set(DataType::FLOAT8_143);
  b.info.set(DataType::FLOAT8_152);
  log2Scale.info.set(DataType::INT32);

  // Throws when log2 not provided
  inputs.insert(0, &a);
  inputs.insert(1, &b);

  BOOST_CHECK_THROW(validateOpFloat8Inputs(&inputs, 2, name), error);
  inputs.clear();

  // Throw on mixing types
  Tensor badB("badB", TensorType::ActGrad, g);
  badB.info.set(DataType::FLOAT);

  runTest(a, badB, log2Scale, "Invalid operand type");

  // Throw on including log2scale for non-float8 types
  Tensor badA("badA", TensorType::ActGrad, g);
  badA.info.set(DataType::FLOAT);

  runTest(badA, badB, log2Scale, "Log2 scale input tensor not accepted");

  // Throw on non-int32 log2scale
  Tensor nonIntLog2Scale("badLog2SCale", TensorType::ActGrad, g);
  nonIntLog2Scale.info.set(DataType::INT16);

  runTest(a, b, nonIntLog2Scale, "Invalid log2 scale input type");

  // Throw on non-scalar log2scale
  Tensor nonScalarLog2Scale("nonScalarLog2Scale", TensorType::ActGrad, g);
  nonScalarLog2Scale.info.set(DataType::INT32, {1, 2, 3});

  runTest(a, b, nonScalarLog2Scale, "must be a scalar tensor");
}

BOOST_AUTO_TEST_CASE(TestOpInputsAreValidPow2ScaledInputs) {
  Ir ir;
  Graph &g = ir.getMainGraph();
  TensorIndexMap inputs;

  Tensor a("a", TensorType::ActGrad, g);
  Tensor b("b", TensorType::ActGrad, g);
  Tensor log2Scale("log2scale", TensorType::ActGrad, g);
  a.info.set(DataType::FLOAT8_143);
  b.info.set(DataType::FLOAT8_152);
  log2Scale.info.set(DataType::INT32);

  // no log2scale
  inputs.insert(0, &a);
  inputs.insert(1, &b);
  BOOST_CHECK(!opInputsAreValidPow2ScaledInputs(&inputs, 2));

  // all inputs are fine
  inputs.insert(2, &log2Scale);
  BOOST_CHECK(opInputsAreValidPow2ScaledInputs(&inputs, 2));
  inputs.clear();

  // mixed types
  Tensor badB("badB", TensorType::ActGrad, g);
  badB.info.set(DataType::FLOAT);

  inputs.insert(0, &a);
  inputs.insert(1, &badB);
  inputs.insert(2, &log2Scale);
  BOOST_CHECK(!opInputsAreValidPow2ScaledInputs(&inputs, 2));
  inputs.clear();

  // log2 scale has bad type
  Tensor nonIntLog2Scale("badLog2SCale", TensorType::ActGrad, g);
  nonIntLog2Scale.info.set(DataType::INT16);

  inputs.insert(0, &a);
  inputs.insert(1, &b);
  inputs.insert(2, &nonIntLog2Scale);
  BOOST_CHECK(!opInputsAreValidPow2ScaledInputs(&inputs, 2));
  inputs.clear();

  // log2 scale is not scalar
  Tensor nonScalarLog2Scale("nonScalarLog2SCale", TensorType::ActGrad, g);
  nonScalarLog2Scale.info.set(DataType::INT32, {1, 2, 3});

  inputs.insert(0, &a);
  inputs.insert(1, &b);
  inputs.insert(2, &nonScalarLog2Scale);
  BOOST_CHECK(!opInputsAreValidPow2ScaledInputs(&inputs, 2));
}
