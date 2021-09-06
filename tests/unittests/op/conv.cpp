// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Test_Op_Conv
#include <boost/test/unit_test.hpp>

#include <popart/ir.hpp>
#include <popart/op/conv.hpp>

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

  // These must be zero
  testParams.kernelTransformation.lowerTruncation = {0, 0};
  testParams.kernelTransformation.upperTruncation = {0, 0};

  // This works
  testParams.kernelTransformation.dilation = {19, 20};

  // These must be zero
  testParams.kernelTransformation.lowerPadding = {0, 0};
  testParams.kernelTransformation.upperPadding = {0, 0};

  // These must be false
  testParams.kernelTransformation.flip = {false, false};

  // These need to work
  testParams.outputTransformation.lowerTruncation = {21, 22};
  testParams.outputTransformation.upperTruncation = {23, 24};
  testParams.outputTransformation.stride          = {24, 25};
  testParams.outputTransformation.lowerPadding    = {26, 27};
  testParams.outputTransformation.upperPadding    = {28, 29};

  std::vector<ConvParameters> paramsVec;
  paramsVec.push_back(testParams);

  logging::warn("About to call op->restoreAttributesFromParams");
  op->restoreAttributesFromParams(paramsVec);

  auto paramsOut = op->getParameters();

  BOOST_TEST(testParams == paramsOut);
};
