// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_Conv
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <popart/ir.hpp>
#include <popart/op/conv.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graph.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/onnxoperators.gen.hpp"
#include "popart/op.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/receptive.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensors.hpp"

using namespace popart;

BOOST_AUTO_TEST_CASE(TestRestoreParams) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;
  auto op      = g.createOp<ConvOp>(Onnx::Operators::Conv_1,
                               settings,
                               std::vector<int64_t>(),
                               std::vector<int64_t>(),
                               std::vector<int64_t>(),
                               1,
                               autoPad,
                               convOpts);

  const TensorInfo dataInfo{DataType::FLOAT, Shape{1, 1, 2, 2}};
  const std::vector<float> dataData(dataInfo.nelms());
  const TensorId dataTensor = "dataTensor";
  g.getTensors().addVarInit(
      dataTensor, dataInfo, dataData.data(), "dataTensorInit");
  op->connectInTensor(0, dataTensor);

  const TensorInfo weightInfo{DataType::FLOAT, Shape{1, 1, 2, 2}};
  const std::vector<float> weightData(weightInfo.nelms());
  const TensorId weightTensor = "weightTensor";
  g.getTensors().addVarInit(
      weightTensor, weightInfo, weightData.data(), "weightTensorInit");
  op->connectInTensor(1, weightTensor);

  ConvParameters testParams;
  testParams.type      = DataType::FLOAT;
  testParams.batchSize = 1;

  // These get overwritten based on the tensor inputs
  testParams.numInChannelsPerGroup  = 1;
  testParams.numOutChannelsPerGroup = 1;

  testParams.numGroups = 1;

  testParams.inputShape  = {2, 2};
  testParams.kernelShape = {2, 2};

  // These need to work
  testParams.inputTransformation.lowerTruncation = {9, 10};
  testParams.inputTransformation.upperTruncation = {11, 12};
  testParams.inputTransformation.dilation        = {13, 14};
  testParams.inputTransformation.lowerPadding    = {15, 16};
  testParams.inputTransformation.upperPadding    = {17, 18};

  // These must be false
  testParams.inputTransformation.flip = {false, false};

  // This works
  testParams.kernelTransformation.dilation = {19, 20};

  // These must be zero
  testParams.kernelTransformation.lowerPadding = {0, 0};
  testParams.kernelTransformation.upperPadding = {0, 0};

  // These must be false
  testParams.kernelTransformation.flip = {false, false};

  // These need to work
  testParams.kernelTransformation.lowerTruncation = {21, 22};
  testParams.kernelTransformation.upperTruncation = {23, 24};

  testParams.outputTransformation.lowerTruncation = {25, 26};
  testParams.outputTransformation.upperTruncation = {27, 28};
  testParams.outputTransformation.stride          = {29, 30};
  testParams.outputTransformation.lowerPadding    = {31, 32};
  testParams.outputTransformation.upperPadding    = {33, 34};

  std::vector<ConvParameters> paramsVec;
  paramsVec.push_back(testParams);

  logging::warn("About to call op->restoreAttributesFromParams");
  op->restoreAttributesFromParams(paramsVec);

  auto paramsOut = op->getParameters();

  BOOST_TEST(testParams == paramsOut);
}

BOOST_AUTO_TEST_CASE(TestFloat8ConvSetup) {
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

  popart::ConvOp convOp(Onnx::Operators::Conv_1,
                        settings,
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        1,
                        autoPad,
                        convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  // deliberately set a non-float8 dtype on the weight tensor to check
  // an error is thrown whenever non-float8 and float8 inputs are mixed
  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  weightTensor.info.set(popart::DataType::FLOAT16, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &log2Scale);
  convOp.output->insert(0, &out);

  // Mixing a float8 tensor with a non-float8 tensor should throw
  expectedMsg =
      logging::format("Invalid operand type: {}", weightTensor.info.dataType());
  BOOST_CHECK(!convOp.isPow2ScaledConv());
  BOOST_CHECK_EXCEPTION(convOp.setup(), error, errorMessageMatches);
  convOp.input->clear();

  // Not providing a log2 scale tensor when the inputs are float8 throws
  weightTensor.info.set(popart::DataType::FLOAT8_143);
  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);

  expectedMsg = "Log2 scale input tensor must be provided";
  BOOST_CHECK(!convOp.isPow2ScaledConv());
  BOOST_CHECK_EXCEPTION(convOp.setup(), error, errorMessageMatches);
  convOp.input->clear();

  // Providing log2 scale for non-FP8 input tensors throws.
  popart::Tensor dataInvalid("dataInvalid", popart::TensorType::ActGrad, g);
  dataInvalid.info.set(popart::DataType::FLOAT, {1, 1, 2, 2});
  popart::Tensor weightsInvalid(
      "weightInvalid", popart::TensorType::ActGrad, g);
  weightsInvalid.info.set(popart::DataType::FLOAT, {1, 1, 2, 2});

  convOp.input->insert(ConvOp::getDataInIndex(), &dataInvalid);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightsInvalid);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &log2Scale);

  expectedMsg = "Log2 scale input tensor not accepted";
  BOOST_CHECK(!convOp.isPow2ScaledConv());
  BOOST_CHECK_EXCEPTION(convOp.setup(), error, errorMessageMatches);
  convOp.input->clear();

  // Proving a non int32 log 2 scale tensor for FP8 inputs throws.
  popart::Tensor invalidLog2Scale(
      "invalidLogScale", popart::TensorType::ActGrad, g);
  invalidLog2Scale.info.set(popart::DataType::FLOAT, {});

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &invalidLog2Scale);

  expectedMsg = logging::format("Invalid log2 scale input type {}",
                                invalidLog2Scale.info.dataType());
  BOOST_CHECK(!convOp.isPow2ScaledConv());
  BOOST_CHECK_EXCEPTION(convOp.setup(), error, errorMessageMatches);
  convOp.input->clear();

  // Non-scalar log2 scale tensor throws.
  popart::Tensor nonScalarLog2Scale(
      "nonScalarLog2Scale", popart::TensorType::ActGrad, g);
  nonScalarLog2Scale.info.set(popart::DataType::INT32, {1, 2, 3});

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &nonScalarLog2Scale);

  expectedMsg = "must be a scalar tensor";
  BOOST_CHECK(!convOp.isPow2ScaledConv());
  BOOST_CHECK_EXCEPTION(convOp.setup(), error, errorMessageMatches);
}

BOOST_AUTO_TEST_CASE(TestFloat8ConvOutputType) {
  // Test that if float8 inputs are used for the conv, the
  // output type is set to float16
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);

  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;
  popart::ConvOp convOp(Onnx::Operators::Conv_1,
                        settings,
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        1,
                        autoPad,
                        convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  weightTensor.info.set(popart::DataType::FLOAT8_152, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &log2Scale);

  convOp.output->insert(0, &out);

  convOp.setup();
  BOOST_CHECK(convOp.outInfo(0).dataType() == DataType::FLOAT16);
  BOOST_CHECK(convOp.outInfo(0).shape().size());
}

BOOST_AUTO_TEST_CASE(TestFloat8ConvPartialsType) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  // deliberately set the partials type to float to force throw on setup.
  auto convAttr                    = Attributes();
  Attributes::String floatPartials = "float";
  convAttr.setAttribute(sPartialsTypeAttribute, floatPartials);

  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);
  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;

  popart::ConvOp convOp(Onnx::Operators::Conv_1,
                        settings,
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        1,
                        autoPad,
                        convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  weightTensor.info.set(popart::DataType::FLOAT8_152, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &log2Scale);

  convOp.output->insert(0, &out);

  // Should throw because the partials type has been
  // set to something other than float16
  BOOST_CHECK_THROW(convOp.setup(), error);
}

BOOST_AUTO_TEST_CASE(TestFloat8ConvDoesNotReturnGradOps) {
  Ir ir;
  Graph &g = ir.getMainGraph();

  auto convAttr = Attributes();
  auto convOpts =
      MultiConvOptions(ir.getSessionOptions().convolutionOptions, convAttr);
  Op::Settings settings(g, "test_conv");
  auto autoPad = AutoPad::VALID;

  popart::ConvOp convOp(Onnx::Operators::Conv_1,
                        settings,
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        std::vector<int64_t>(),
                        1,
                        autoPad,
                        convOpts);
  Tensor dataTensor("data", popart::TensorType::ActGrad, g);
  dataTensor.info.set(popart::DataType::FLOAT8_143, {1, 1, 2, 2});

  Tensor weightTensor("weight", popart::TensorType::ActGrad, g);
  weightTensor.info.set(popart::DataType::FLOAT8_152, {1, 1, 2, 2});

  Tensor log2Scale("log2Scale", popart::TensorType::ActGrad, g);
  log2Scale.info.set(popart::DataType::INT32, {});

  Tensor out("outputTensor", popart::TensorType::ActGrad, g);

  convOp.input->insert(ConvOp::getDataInIndex(), &dataTensor);
  convOp.input->insert(ConvOp::getWeightsInIndex(), &weightTensor);
  convOp.input->insert(ConvOp::getLog2ScaleInIndex(), &log2Scale);

  convOp.output->insert(0, &out);

  BOOST_CHECK_THROW(convOp.getGradOps(), error);
}
