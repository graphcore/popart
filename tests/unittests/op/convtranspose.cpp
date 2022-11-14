// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConvTransposePatternTests
#include <boost/test/unit_test.hpp>
#include <map>

#include "popart/graph.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/onnxoperators.gen.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/convtranspose.hpp"
#include "popart/tensorindex.hpp"

namespace popart {
class error;
}

using namespace popart;

BOOST_AUTO_TEST_CASE(TestConvTransposeThrowsOnInvalidChannels) {
  // Deliberately set up some conv values that will throw due to
  // channel dimensions not matching on the input and weights
  int inChannel    = 10;
  int outChannel   = 20;
  int inputHeight  = 50;
  int inputWidth   = 50;
  int weightHeight = 3;
  int weightWidth  = 3;

  Shape inputShape  = {1, inChannel, inputHeight, inputWidth};
  Shape weightShape = {outChannel, inChannel, weightHeight, weightWidth};

  Ir ir;
  Graph &graph = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(graph, "test_convtranspose");
  auto autoPad = AutoPad::NOTSET;

  Tensor dataTensor("data", popart::TensorType::ActGrad, graph);
  dataTensor.info.set(popart::DataType::FLOAT, inputShape);

  Tensor weightTensor("weight", popart::TensorType::ActGrad, graph);
  weightTensor.info.set(popart::DataType::FLOAT, weightShape);

  Tensor out("output", popart::TensorType::ActGrad, graph);

  // Create a conv transpose op
  popart::ConvTransposeOp convTransposeOp(Onnx::Operators::Conv_1,
                                          settings,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          1,
                                          autoPad,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          convOpts);

  convTransposeOp.input->insert(0, &dataTensor);
  convTransposeOp.input->insert(1, &weightTensor);
  convTransposeOp.output->insert(0, &out);

  auto checkError = [&inputShape](popart::error const &error) {
    auto expected = logging::format(
        "Unexpected number of channels in the input tensor: {}", inputShape[1]);
    return std::string(error.what()).find(expected) != std::string::npos;
  };
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, checkError);
}

BOOST_AUTO_TEST_CASE(ConvTranpose_Float8Setup) {
  std::string expectedMsg;
  auto errorMessageMatches = [&expectedMsg](popart::error const &error) {
    return std::string(error.what()).find(expectedMsg) != std::string::npos;
  };

  Ir ir;
  Graph &g = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;
  popart::ConvTransposeOp convTransposeOp(Onnx::Operators::Conv_1,
                                          settings,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          1,
                                          autoPad,
                                          std::vector<int64_t>(),
                                          std::vector<int64_t>(),
                                          convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  // deliberately set a non-float8 dtype on the weight tensor to check
  // an error is thrown whenever non-float8 and float8 inputs are mixed
  weightTensor.info.set(popart::DataType::FLOAT16, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convTransposeOp.input->insert(ConvTransposeOp::getInIndex(), &dataTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getWeightsInIndex(),
                                &weightTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getLog2ScaleInIndex(),
                                &log2Scale);

  convTransposeOp.output->insert(0, &out);

  expectedMsg =
      logging::format("Invalid operand type: {}", weightTensor.info.dataType());
  BOOST_CHECK(!convTransposeOp.isPow2ScaledConvTranspose());
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, errorMessageMatches);

  weightTensor.info.set(popart::DataType::FLOAT8_143);

  convTransposeOp.input->clear();
  convTransposeOp.input->insert(ConvTransposeOp::getInIndex(), &dataTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getWeightsInIndex(),
                                &weightTensor);

  expectedMsg = "Log2 scale input tensor must be provided";
  BOOST_CHECK(!convTransposeOp.isPow2ScaledConvTranspose());
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, errorMessageMatches);

  // Providing log2 scale for non-FP8 input tensors throws.
  popart::Tensor dataInvalid("dataInvalid", popart::TensorType::ActGrad, g);
  dataInvalid.info.set(popart::DataType::FLOAT, {1, 1, 2, 2});
  popart::Tensor weightsInvalid(
      "weightInvalid", popart::TensorType::ActGrad, g);
  weightsInvalid.info.set(popart::DataType::FLOAT, {1, 1, 2, 2});

  convTransposeOp.input->clear();
  convTransposeOp.input->insert(ConvTransposeOp::getInIndex(), &dataInvalid);
  convTransposeOp.input->insert(ConvTransposeOp::getWeightsInIndex(),
                                &weightsInvalid);
  convTransposeOp.input->insert(ConvTransposeOp::getLog2ScaleInIndex(),
                                &log2Scale);

  expectedMsg = "Log2 scale input tensor not accepted";
  BOOST_CHECK(!convTransposeOp.isPow2ScaledConvTranspose());
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, errorMessageMatches);

  // Proving a non int32 log 2 scale tensor for FP8 inputs throws.
  popart::Tensor invalidLog2Scale(
      "invalidLogScale", popart::TensorType::ActGrad, g);
  invalidLog2Scale.info.set(popart::DataType::FLOAT, {});

  convTransposeOp.input->clear();
  convTransposeOp.input->insert(ConvTransposeOp::getInIndex(), &dataTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getWeightsInIndex(),
                                &weightTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getLog2ScaleInIndex(),
                                &invalidLog2Scale);

  expectedMsg = logging::format("Invalid log2 scale input type {}",
                                invalidLog2Scale.info.dataType());
  BOOST_CHECK(!convTransposeOp.isPow2ScaledConvTranspose());
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, errorMessageMatches);

  // Non-scalar log2 scale tensor throws.
  popart::Tensor nonScalarLog2Scale(
      "nonScalarLog2Scale", popart::TensorType::ActGrad, g);
  nonScalarLog2Scale.info.set(popart::DataType::INT32, {1, 2, 3});

  convTransposeOp.input->clear();
  convTransposeOp.input->insert(ConvTransposeOp::getInIndex(), &dataTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getWeightsInIndex(),
                                &weightTensor);
  convTransposeOp.input->insert(ConvTransposeOp::getLog2ScaleInIndex(),
                                &nonScalarLog2Scale);

  expectedMsg = "must be a scalar tensor";
  BOOST_CHECK(!convTransposeOp.isPow2ScaledConvTranspose());
  BOOST_CHECK_EXCEPTION(convTransposeOp.setup(), error, errorMessageMatches);
}

BOOST_AUTO_TEST_CASE(TestFloat8ConvTransposeOutputType) {
  // Test that if float8 inputs are used for the conv, transposed the
  // output type is set to float16
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;
  popart::ConvTransposeOp convOp(Onnx::Operators::Conv_1,
                                 settings,
                                 std::vector<int64_t>(),
                                 std::vector<int64_t>(),
                                 std::vector<int64_t>(),
                                 1,
                                 autoPad,
                                 std::vector<int64_t>(),
                                 std::vector<int64_t>(),
                                 convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  weightTensor.info.set(popart::DataType::FLOAT8_152, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convOp.input->insert(ConvTransposeOp::getInIndex(), &dataTensor);
  convOp.input->insert(ConvTransposeOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvTransposeOp::getLog2ScaleInIndex(), &log2Scale);

  convOp.output->insert(0, &out);

  convOp.setup();
  BOOST_CHECK(convOp.outInfo(ConvTransposeOp::getOutIndex()).dataType() ==
              DataType::FLOAT16);
}
