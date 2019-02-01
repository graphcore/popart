#define BOOST_TEST_MODULE ViewChangingTest

#include <boost/test/unit_test.hpp>
#include <map>
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
#include <poponnx/tensors.hpp>

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

// more constexpr transpose tests in fp16_test.py
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

  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();
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

  int expectedOutput[10] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
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

  auto cpuDevice =
      poponnx::DeviceManager::createDeviceManager().createCpuDevice();
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

  int expectedOutput[24] = {1, 5, 9,  13, 17, 21, 2, 6, 10, 14, 18, 22,
                            3, 7, 11, 15, 19, 23, 4, 8, 12, 16, 20, 24};

  BOOST_CHECK(std::equal(&expectedOutput[0],
                         &expectedOutput[24],
                         &rawOutputData[0]) == true);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Add1) {
  // Testing ConstExpr folding on broadcast adds of
  // initializers and constants

  // The compute graph :
  //
  // w0 --|
  //      |--[bc-add]-- a0 --|
  // w1 --|                  |
  //                         |--[bc-add]-- a2 -|
  // c0 --|                  |                 |
  //      |--[bc-add]-- a1 --|                 |-- [matmul] -- o
  // c1 --|                                    |
  //                                           |
  // i1 ---------------------------------------|
  //

  // weights
  TensorInfo w0Shape{"FLOAT", std::vector<int64_t>{1, 3}};
  float w0Vals[1 * 3]  = {0};
  ConstVoidData w0Data = {w0Vals, w0Shape};

  TensorInfo w1Shape{"FLOAT", std::vector<int64_t>{3, 3}};
  float w1Vals[3 * 3]  = {1};
  ConstVoidData w1Data = {w1Vals, w1Shape};

  // consts
  TensorInfo c0Shape{"FLOAT", std::vector<int64_t>{1, 3}};
  float c0Vals[1 * 3]  = {2};
  ConstVoidData c0Data = {c0Vals, c0Shape};

  TensorInfo c1Shape{"FLOAT", std::vector<int64_t>{1}};
  float c1Vals[1]      = {3};
  ConstVoidData c1Data = {c1Vals, c1Shape};

  // input
  TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{3, 4}};

  // Build an onnx model
  auto builder = Builder::create();

  auto w0Id = builder->addInitializedInputTensor(w0Data);
  auto w1Id = builder->addInitializedInputTensor(w1Data);
  auto a0   = builder->add({w0Id, w1Id}, "a0");

  auto c0Id = builder->constant(c0Data, "c0Data");
  auto c1Id = builder->constant(c1Data, "c1Data");
  auto a1   = builder->add({c0Id, c1Id}, "a1");

  auto a2      = builder->add({a0, a1}, "a2");
  auto inputId = builder->addInputTensor(inputInfo);
  auto outId   = builder->matmul({a2, inputId});
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
              {}, // no loss
              {}, // no optimizer
              {}, // no SessionOptions
              Patterns({PatternType::POSTNREPL})});

  // Check that the Add Op is has been removed from the IR
  // by ConstExpr folding
  // TODO: this test will give a false pass if the model is using
  // a newer opset. Fix when T6274 is complete
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 0);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Add2) {
  // Testing ConstExpr folding on an input tensor whose consumer
  // has another input that cannot be removed by ConstExpr folding

  // Model from the builder:
  //
  // v0 ----|
  //        |--[add]-- a0 --|
  //        |               |
  // c0 --|-|               |--[add]-- o
  //      |                 |
  //      |--[add]---- a1 --|
  // c1 --|
  //
  // Expected outcome after ConstExpr folding :
  //
  // v0 ---|
  //       |--[add]-- a0 --|
  // c0 ---|               |--[add]-- o
  //                       |
  // a1 -------------------|
  //

  // consts
  TensorInfo c0Shape{"FLOAT", std::vector<int64_t>{2, 2}};
  float c0Vals[2 * 2]  = {2};
  ConstVoidData c0Data = {c0Vals, c0Shape};

  TensorInfo c1Shape{"FLOAT", std::vector<int64_t>{2, 2}};
  float c1Vals[2 * 2]  = {3};
  ConstVoidData c1Data = {c1Vals, c1Shape};

  // input
  TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2, 2}};

  // Build an onnx model
  auto builder = Builder::create();

  auto v0Id = builder->addInputTensor(inputInfo);
  auto c0Id = builder->constant(c0Data, "c0Data");
  auto c1Id = builder->constant(c1Data, "c1Data");

  auto a0 = builder->add({v0Id, c0Id}, "a0");
  auto a1 = builder->add({c0Id, c1Id}, "a1");

  auto o = builder->add({a0, a1}, "o");
  builder->addOutputTensor(o);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("ALL");
  auto dataFlow = DataFlow(1, {{o, art}});

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {}, // no loss
              {}, // no optimizer
              {}, // no SessionOptions
              Patterns({PatternType::POSTNREPL})});

  // Check that the producer of a1 Add Op is has been removed from the IR
  // by ConstExpr folding
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 2);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Slice0) {
  // clang-format off
  //
  // [Constant] -> (d) -> [Slice] -> (h0)
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // where d is constant, should become
  //
  // {(h0), (in1)} -> [Add] -> (*)
  //
  // clang-format on

  std::vector<float> raw_const_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  ConstVoidData const_data = {raw_const_data.data(), {"FLOAT", Shape{2, 4}}};

  TensorInfo in_info{"FLOAT", Shape{1}};

  auto builder    = Builder::create();
  auto const_node = builder->constant(const_data, "const_data");
  auto slice_node = builder->slice({const_node}, {0, 1}, {1, 0}, {2, 3});
  auto in_id      = builder->addInputTensor(in_info);
  auto out_id     = builder->add({slice_node, in_id});
  builder->addOutputTensor(out_id);

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("ALL");
  auto data_flow = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  std::vector<Loss *> losses{new L1Loss(out_id, "l1LossVal", 0.1)};

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              losses,
              &optimizer,
              {}, // no SessionOptions
              Patterns({})});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the Slice Op is not present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
  // 3) that the shape of the output tensor is as specified.
  Shape ref_shape{1, 3};
  BOOST_CHECK(ir.getTensors().get(out_id)->info.shape() == ref_shape);
}
