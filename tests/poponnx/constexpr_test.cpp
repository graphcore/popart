#define BOOST_TEST_MODULE ViewChangingTest

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
// needed for getting model from string
#include <poponnx/filereader.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
// needed for ConstVoidData
#include <poponnx/devicemanager.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

BOOST_AUTO_TEST_CASE(ConstExprTest_Add0) {

  // The compute graph :
  //
  // data  -----------------------------|
  //                                    |
  //                                    |
  //                                    |- RESHAPE ---> output
  //                                    |
  // shape0 -------|                    |
  //               |                    |
  //               |- ADD - outshape ---|
  //               |
  // shape1 -------|

  // We will reshape a tensor from rank-4:
  Shape inShape = {2, 5, 3, 4};
  // to rank-2: {10, 12},
  // Note above that the total number elements of the tensor remains 120

  // where the output shape {10, 12} will be the sum of two tensors,
  // 1)
  Shape shape0 = {7, 4};
  // 2)
  Shape shape1 = {3, 8};

  Shape outShapeSize = {static_cast<int64_t>(shape0.size())};
  TensorInfo inInfo{"FLOAT", inShape};

  ConstVoidData out0ShapeData = {shape0.data(), {"INT64", outShapeSize}};
  ConstVoidData out1ShapeData = {shape1.data(), {"INT64", outShapeSize}};

  // Build an onnx model
  auto builder = Builder::create();
  // The two fixed-point tensors which are Constants
  auto shape0Id   = builder->constant(out0ShapeData, "out0ShapeData");
  auto shape1Id   = builder->constant(out1ShapeData, "out1ShapeData");
  auto inId       = builder->addInputTensor(inInfo);
  auto outShapeId = builder->add({shape0Id, shape1Id});
  auto outId      = builder->reshape({inId, outShapeId});
  builder->addOutputTensor(outId);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(outId, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {}, // no SessionOptions
              Patterns({PatternType::POSTNREPL})});

  // Check the ir
  // 1) that the Reshape Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Reshape).size() == 1);
  // 2) that the shape of the output tensor is as specified.
  Shape outShape;
  for (int i = 0; i < outShapeSize[0]; ++i) {
    outShape.push_back(shape0[i] + shape1[i]);
  }
  BOOST_CHECK(ir.getTensors().get(outId)->info.shape() == outShape);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_AddCastMatMul) {

  /*********

   The compute graph

-> data, float (7,5) ------------------------|
                                             |- MATMUL --- out:, float (7,3)
                                              \
                                               \
                                                \
-> i0, int32, (5,1)-|                            \
                    |                             \
                    |                              \
                    |- ADD --| CAST -- float (5,3) -|
-> i1, int32 (1,3)--|

  ***********/

  // Build the onnx model described
  // in the schematic above
  int64_t M = 7;
  int64_t K = 5;
  int64_t N = 3;

  std::vector<int64_t> outshape{M, N};
  std::vector<int64_t> weights_shape{K, N};
  std::vector<int64_t> data_shape{M, K};

  std::vector<int> i0(K);
  std::iota(i0.begin(), i0.end(), 1);
  std::vector<int> i1(N);
  std::iota(i1.begin(), i1.end(), 1);
  TensorInfo dataInfo{"FLOAT", std::vector<int64_t>{M, K}};
  ConstVoidData i0cv = {i0.data(), {"INT32", std::vector<int64_t>{K, 1}}};
  ConstVoidData i1cv = {i1.data(), {"INT32", std::vector<int64_t>{1, N}}};
  auto builder       = Builder::create();
  // The two fixed-point tensors which are added together are Constants
  auto i0Id   = builder->constant(i0cv, "i0cv");
  auto i1Id   = builder->constant(i1cv, "i1cv");
  auto dataId = builder->addInputTensor(dataInfo);
  auto i01Id  = builder->add({i0Id, i1Id});
  auto castId = builder->cast({i01Id}, DataType::FLOAT);
  auto outId  = builder->matmul({dataId, castId});
  builder->addOutputTensor(outId);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(outId, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              losses,
              &optimizer,
              {}, // no SessionOptions
              Patterns({PatternType::POSTNREPL})});

  // Check the ir
  // 1) The Matmul Op is present,
  auto matmuls = ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul);
  BOOST_CHECK(matmuls.size() == 1);
  BOOST_CHECK(matmuls[0]->input->hasIndex(1));
  auto weights = matmuls[0]->input->tensor(1);

  // 2) The shape of the output is correct
  BOOST_CHECK(ir.getTensors().get(outId)->info.shape() == outshape);

  // 3) The weights inputs to the matmul are correct,
  BOOST_CHECK(weights->info.shape() == weights_shape);
  BOOST_CHECK(weights->tensorType() == TensorType::Const);
  BOOST_CHECK(weights->hasTensorData() == true);
  auto tensorWData = weights->tensorData()->data();
  auto floatWData  = reinterpret_cast<float *>(tensorWData);
  // The weights were formed as the outer product of two
  // vectors, {1...K} and {1...N}. We therefore expect the very
  // last element of the weights tensor to be K + N
  BOOST_CHECK(floatWData[K * N - 1] == static_cast<float>(K + N));
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Transpose1) {

  Shape inShape = {2, 5};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {5, 2};
  std::vector<int> rawConstInputData(5 * 2);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  poponnx::ArrayWrapper<int> constData({5, 2}, rawConstInputData.data());

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constShape}};

  // Build an onnx model
  auto builder = Builder::create();

  auto constId = builder->constant(constShapeData, "out0ShapeData");
  auto inId    = builder->addInputTensor(inInfo);

  auto outShapeId = builder->transpose({constId}, {});
  auto out        = builder->add({outShapeId, inId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{out, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out, "l1LossVal", 0.1)};

  auto session = poponnx::Session::createFromOnnxModel(
      proto,
      dataFlow,
      poponnx::InputShapeInfo(),
      losses,
      &optimizer,
      {},
      poponnx::Patterns({poponnx::PatternType::POSTNREPL}));

  auto cpuDevice = poponnx::DeviceManager::getDeviceManager().createCpuDevice();
  session->setDevice(*cpuDevice);

  // prepare the anchors
  int rawOutputData[10];
  poponnx::ArrayWrapper<int> outData({2, 5}, rawOutputData);

  std::map<poponnx::TensorId, poponnx::Array &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  int rawInputData[10] = {
      0,
  };
  poponnx::ArrayWrapper<int> inData({2, 5}, rawInputData);
  std::map<poponnx::TensorId, poponnx::Array &> inputs = {{inId, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->infer(stepio);

  poponnx::logging::ir::err("const : {}", constData);
  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);

  int expectedOutput[10] = {1, 6, 2, 7, 3, 8, 4, 9, 5, 10};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[10],
                         &rawOutputData[0]) == true);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Transpose2) {

  Shape inShape = {4, 2, 3};
  TensorInfo inInfo{"INT32", inShape};

  Shape constShape = {2, 3, 4};
  std::vector<int> rawConstInputData(4 * 2 * 3);
  std::iota(rawConstInputData.begin(), rawConstInputData.end(), 1);

  poponnx::ArrayWrapper<int> constData({2, 3, 4}, rawConstInputData.data());

  ConstVoidData constShapeData = {rawConstInputData.data(),
                                  {"INT32", constShape}};

  // Build an onnx model
  auto builder = Builder::create();

  auto constId = builder->constant(constShapeData, "constShapeData");
  auto inId    = builder->addInputTensor(inInfo);

  auto outShapeId = builder->transpose({constId}, {2, 0, 1});
  auto out        = builder->add({outShapeId, inId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto dataFlow  = DataFlow(1, {{out, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out, "l1LossVal", 0.1)};

  auto session = poponnx::Session::createFromOnnxModel(
      proto,
      dataFlow,
      poponnx::InputShapeInfo(),
      losses,
      &optimizer,
      {},
      poponnx::Patterns({poponnx::PatternType::POSTNREPL}));

  auto cpuDevice = poponnx::DeviceManager::getDeviceManager().createCpuDevice();
  session->setDevice(*cpuDevice);

  // prepare the anchors
  int rawOutputData[24];
  poponnx::ArrayWrapper<int> outData({4, 2, 3}, rawOutputData);

  std::map<poponnx::TensorId, poponnx::Array &> anchors = {
      {out, outData},
  };

  session->prepareDevice();

  // prepare the inputs
  int rawInputData[24] = {
      0,
  };
  poponnx::ArrayWrapper<int> inData({4, 2, 3}, rawInputData);
  std::map<poponnx::TensorId, poponnx::Array &> inputs = {{inId, inData}};

  poponnx::StepIO stepio(inputs, anchors);

  session->infer(stepio);

  poponnx::logging::ir::err("const : {}", constData);
  poponnx::logging::ir::err("input : {}", inData);
  poponnx::logging::ir::err("output : {}", outData);

  int expectedOutput[24] = {1, 7,  13, 19, 2, 8,  14, 20, 3, 9,  15, 21,
                            4, 10, 16, 22, 5, 11, 17, 23, 6, 12, 18, 24};
  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[24],
                         &rawOutputData[0]) == true);
}