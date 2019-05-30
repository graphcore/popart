#define BOOST_TEST_MODULE ViewChangingTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

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
  auto builder    = Builder::create();
  auto aiOnnx     = builder->aiOnnxOpset9();
  auto newShapeId = aiOnnx.constant(outShapeData, "outShapeData");
  auto inId       = builder->addInputTensor(inInfo);
  auto outId      = aiOnnx.reshape({inId, newShapeId});
  builder->addOutputTensor(outId);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(outId, "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {}, // no SessionOptions
              Patterns({PreAliasPatternType::POSTNREPL})});

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
  auto newShapeId            = builder->addInitializedInputTensor(outShapeData);

  auto inId  = builder->addInputTensor(inInfo);
  auto outId = aiOnnx.reshape({inId, newShapeId});
  builder->addOutputTensor(outId);
  auto proto = builder->getModelProto();

  // The new Shape Tensor is not a weight initializer, and should
  // therefore be converted to the output of an ONNX Constant.
  // We use this convert-all-non-float function to do the conversion.
  GraphTransformer graph_transformer(proto);
  graph_transformer.convertAllFixedPointInitializersToConstants();
  proto = graph_transformer.getModelProto();

  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("ALL");
  auto dataFlow   = DataFlow(1, {{outId, art}});
  auto optimizer  = ConstSGD(0.01);
  std::vector<Loss *> losses{
      new L1Loss(outId, "l1LossVal", 0.1, ReductionType::SUM)};
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              *cpuDevice,
              {},
              Patterns({PreAliasPatternType::POSTNREPL})});
  BOOST_CHECK(ir.opsOfType(Onnx::Operators::Reshape_5).size() == 1);
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outShape);
}
