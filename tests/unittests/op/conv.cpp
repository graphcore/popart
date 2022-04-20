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
#include "popart/op.hpp"
#include "popart/op/convbase.hpp"
#include "popart/op/receptive.hpp"
#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensordebuginfo.hpp"
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
