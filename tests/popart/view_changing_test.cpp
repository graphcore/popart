// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ViewChangingTest

#include <memory>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(ViewChangingTest_Reshape0) {

  // We will reshape a tensor from rank-4:
  Shape inShape = {2, 5, 3, 4};
  // to rank-2:
  Shape outShape = {10, 12};
  // Note above that the total number elements of the tensor remains 120

  Shape outShapeSize = {static_cast<int64_t>(outShape.size())};
  TensorInfo inInfo{"FLOAT", inShape};
  ConstVoidData outShapeData = {outShape.data(), {"INT64", outShapeSize}};

  // Build an onnx model
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  auto newShapeId  = aiOnnx.constant(outShapeData, "outShapeData");
  auto inId        = builder->addInputTensor(inInfo);
  auto outId       = aiOnnx.reshape({inId, newShapeId});
  auto lossId      = aiGraphcore.l1loss({outId}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              lossId,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns({PreAliasPatternType::PostNRepl})});

  // Check the ir
  // 1) that the Reshape Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Reshape).size() == 1);
  // 2) that the shape of the output tensor is as specified.
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outShape);
}

// as in ViewChangingTest_Reshape0, but uses an
// initializer instead of a constant for the shape
BOOST_AUTO_TEST_CASE(ViewChangingTest_Reshape_Initializer) {
  Shape inShape      = {2, 5, 3, 4};
  Shape outShape     = {10, 12};
  Shape outShapeSize = {static_cast<int64_t>(outShape.size())};
  TensorInfo inInfo{"FLOAT", inShape};
  ConstVoidData outShapeData = {outShape.data(), {"INT64", outShapeSize}};
  auto builder               = Builder::create();
  auto aiOnnx                = builder->aiOnnxOpset9();
  auto aiGraphcore           = builder->aiGraphcoreOpset1();
  auto newShapeId            = builder->addInitializedInputTensor(outShapeData);

  auto inId   = builder->addInputTensor(inInfo);
  auto outId  = aiOnnx.reshape({inId, newShapeId});
  auto lossId = aiGraphcore.l1loss({outId}, 0.1);

  auto proto = builder->getModelProto();

  // The new Shape Tensor is not a weight initializer, and should
  // therefore be converted to the output of an ONNX Constant.
  // We use this convert-all-non-float function to do the conversion.
  GraphTransformer graph_transformer(proto);
  graph_transformer.convertAllFixedPointInitializersToConstants();
  proto = graph_transformer.getModelProto();

  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto dataFlow   = DataFlow(1, {{outId, art}});
  auto optimizer  = ConstSGD(0.01);
  auto device     = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              lossId,
              &optimizer,
              *device,
              {},
              Patterns({PreAliasPatternType::PostNRepl})});
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Reshape_5).size() == 1);
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outShape);
}
