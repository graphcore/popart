// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprAddTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/session.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <math.h>

using namespace popart;

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
  auto aiOnnx  = builder->aiOnnxOpset9();
  // The two fixed-point tensors which are Constants
  auto shape0Id   = aiOnnx.constant(out0ShapeData, "out0ShapeData");
  auto shape1Id   = aiOnnx.constant(out1ShapeData, "out1ShapeData");
  auto inId       = builder->addInputTensor(inInfo);
  auto outShapeId = aiOnnx.add({shape0Id, shape1Id});
  auto outId      = aiOnnx.reshape({inId, outShapeId});
  auto l1         = builder->aiGraphcoreOpset1().l1loss({outId}, 0.1);

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
              l1,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) that the Reshape Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Reshape).size() == 1);
  // 2) that the shape of the output tensor is as specified.
  Shape outShape;
  for (int i = 0; i < outShapeSize[0]; ++i) {
    outShape.push_back(shape0[i] + shape1[i]);
  }
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outShape);
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
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto w0Id = builder->addInitializedInputTensor(w0Data);
  auto w1Id = builder->addInitializedInputTensor(w1Data);
  auto a0   = aiOnnx.add({w0Id, w1Id}, "a0");

  auto c0Id = aiOnnx.constant(c0Data, "c0Data");
  auto c1Id = aiOnnx.constant(c1Data, "c1Data");
  auto a1   = aiOnnx.add({c0Id, c1Id}, "a1");

  auto a2      = aiOnnx.add({a0, a1}, "a2");
  auto inputId = builder->addInputTensor(inputInfo);
  auto outId   = aiOnnx.matmul({a2, inputId});
  auto l1      = builder->aiGraphcoreOpset1().l1loss({outId}, 0.1);

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
              {}, // no loss
              {}, // no optimizer
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

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
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto v0Id = builder->addInputTensor(inputInfo);
  auto c0Id = aiOnnx.constant(c0Data, "c0Data");
  auto c1Id = aiOnnx.constant(c1Data, "c1Data");

  auto a0 = aiOnnx.add({v0Id, c0Id}, "a0");
  auto a1 = aiOnnx.add({c0Id, c1Id}, "a1");

  auto o = aiOnnx.add({a0, a1}, "o");
  builder->addOutputTensor(o);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{o, art}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {}, // no loss
              {}, // no optimizer
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check that the producer of a1 Add Op is has been removed from the IR
  // by ConstExpr folding
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 2);
}

template <typename T> void ConstExprTest_Add_Type(std::string type) {

  // The compute graph :
  //
  // data  -----------------------------|
  //                                    |
  //                                    |
  //                                    |- ADD ---> output
  //                                    |
  // shape0 -------|                    |
  //               |                    |
  //               |- ADD - outshape ---|
  //               |
  // shape1 -------|

  Shape inShape = {2, 2};

  T input1_raw[] = {(T)1.2f, 2, 3, 4};
  std::vector<T> input1(input1_raw, std::end(input1_raw));

  T input2_raw[] = {(T)1.7f, 2, 3, 4};
  std::vector<T> input2(input2_raw, std::end(input2_raw));

  Shape outShapeSize = {2, 2};
  TensorInfo inInfo{type, inShape};

  ConstVoidData out0ShapeData = {input1.data(), {type, outShapeSize}};
  ConstVoidData out1ShapeData = {input2.data(), {type, outShapeSize}};

  T output_raw[4];
  popart::NDArrayWrapper<T> output(output_raw, {2, 2});

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  // The two fixed-point tensors which are Constants
  auto shape0Id   = aiOnnx.constant(out0ShapeData, "out0ShapeData");
  auto shape1Id   = aiOnnx.constant(out1ShapeData, "out1ShapeData");
  auto inId       = builder->addInputTensor(inInfo);
  auto outShapeId = aiOnnx.add({shape0Id, shape1Id});
  auto outId      = aiOnnx.add({inId, outShapeId});
  builder->addOutputTensor(outId);

  std::map<popart::TensorId, popart::IArray &> anchors = {{outId, output}};

  auto proto = builder->getModelProto();

  auto art      = AnchorReturnType("All");
  auto dataFlow = DataFlow(1, {{outId, art}});

  auto device = popart::createTestDevice(TEST_TARGET);

  auto session = popart::InferenceSession::createFromOnnxModel(
      proto,
      dataFlow,
      device,
      InputShapeInfo(),
      {}, // no SessionOptions
      Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false));

  T rawInputData[4] = {(T)1.1f, 2, 3, 4};
  popart::NDArrayWrapper<T> inData(rawInputData, {2, 2});
  std::map<popart::TensorId, popart::IArray &> inputs = {{inId, inData}};

  session->prepareDevice();
  popart::StepIO stepio(inputs, anchors);
  session->run(stepio);

  // Check the ir
  popart::logging::ir::err("input1 : {}", input1[0]);
  popart::logging::ir::err("input2 : {}", input2[0]);
  popart::logging::ir::err("indata : {}", inData[0]);
  popart::logging::ir::err("output : {}", output[0]);

  BOOST_CHECK((input1[0] + input2[0] + inData[0]) == output[0]);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Div0) {

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
  Shape inShape = {2, 2, 2};
  // to rank-2: {2, 4},
  // Note above that the total number elements of the tensor remains 120

  // where the output shape {2, 4} will be the sum of two tensors,
  // 1)
  Shape shape0 = {4, 12};
  // 2)
  Shape shape1 = {2, 3};

  Shape outShapeSize = {static_cast<int64_t>(shape0.size())};
  TensorInfo inInfo{"FLOAT", inShape};

  ConstVoidData out0ShapeData = {shape0.data(), {"INT64", outShapeSize}};
  ConstVoidData out1ShapeData = {shape1.data(), {"INT64", outShapeSize}};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  // The two fixed-point tensors which are Constants
  auto shape0Id   = aiOnnx.constant(out0ShapeData, "out0ShapeData");
  auto shape1Id   = aiOnnx.constant(out1ShapeData, "out1ShapeData");
  auto inId       = builder->addInputTensor(inInfo);
  auto outShapeId = aiOnnx.div({shape0Id, shape1Id});
  auto outId      = aiOnnx.reshape({inId, outShapeId});
  auto l1         = builder->aiGraphcoreOpset1().l1loss({outId}, 0.1);

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
              l1,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) that the Reshape Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Reshape).size() == 1);
  // 2) that the shape of the output tensor is as specified.
  Shape outShape;
  for (int i = 0; i < outShapeSize[0]; ++i) {
    outShape.push_back(shape0[i] / shape1[i]);
  }
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outShape);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Add_Types) {
  // ConstExprTest_Add_Type<uint32_t>("UINT32");
  // ConstExprTest_Add_Type<uint64_t>("UINT64");
  ConstExprTest_Add_Type<int32_t>("INT32");
  // ConstExprTest_Add_Type<int64_t>("INT64");
  ConstExprTest_Add_Type<popart::float16_t>("FLOAT16");
  ConstExprTest_Add_Type<float>("FLOAT");
  // ConstExprTest_Add_Type("DOUBLE");
}

template <typename T> std::string getTypeString();

template <> std::string getTypeString<int64_t>() { return "INT64"; }
template <> std::string getTypeString<float16_t>() { return "FLOAT16"; }
template <> std::string getTypeString<float_t>() { return "FLOAT"; }

template <typename T>
void ConstExprTest_Elementwise_Test(
    std::vector<T> in0,
    std::vector<T> in1,
    std::vector<T> output,
    std::function<TensorId(std::unique_ptr<popart::Builder> &builder,
                           const std::vector<TensorId> &args)> elementWiseFn) {

  // The compute graph :
  //
  // data  -----------------------------|
  //                                    |
  //                                    |
  //                                    |- ADD ---> output
  //                                    |
  // shape0 -------|                    |
  //               |                    |
  //               |- OP - outshape ----|
  //               |
  // shape1 -------|

  std::vector<T> data = {42, 42};
  Shape dataShape     = {static_cast<int64_t>(data.size())};
  TensorInfo dataInfo{getTypeString<T>(), dataShape};

  Shape in0Shape        = {static_cast<int64_t>(in0.size())};
  ConstVoidData in0Data = {in0.data(), {getTypeString<T>(), in0Shape}};

  Shape in1Shape        = {static_cast<int64_t>(in1.size())};
  ConstVoidData in1Data = {in1.data(), {getTypeString<T>(), in1Shape}};

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset10();
  // The two fixed-point tensors which are Constants
  auto in0Id      = aiOnnx.constant(in0Data, "in0Data");
  auto in1Id      = aiOnnx.constant(in1Data, "in1Data");
  auto dataId     = builder->addInputTensor(dataInfo);
  auto outShapeId = elementWiseFn(builder, {in0Id, in1Id});

  auto outId = aiOnnx.add({dataId, outShapeId});
  auto l1    = builder->aiGraphcoreOpset1().l1loss({outId}, 0.1);

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
              l1,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the value of the output tensor is as specified.
  T *p = static_cast<T *>(
      ir.getMainGraphTensors().get(outShapeId)->tensorData()->data());
  for (int i = 0; i < output.size(); ++i) {
    BOOST_CHECK(p[i] == output[i]);
  }
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Elementwise) {
  ConstExprTest_Elementwise_Test<int64_t>(
      {4, 12},
      {2, 3},
      {2, 4},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().div(args);
      });

  ConstExprTest_Elementwise_Test<int64_t>(
      {4, 12},
      {2}, // Will be broadcast to {2, 2}
      {2, 6},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().div(args);
      });

  ConstExprTest_Elementwise_Test<float16_t>(
      {8.0, 18.0},
      {4.0, 3.0},
      {2.0, 6.0},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().div(args);
      });

  ConstExprTest_Elementwise_Test<float_t>(
      {10.0, 21.0},
      {4.0, 5.0},
      {2.5, 4.2},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().div(args);
      });

  ConstExprTest_Elementwise_Test<int64_t>(
      {4, 12},
      {2, 3},
      {6, 15},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().add(args);
      });

  ConstExprTest_Elementwise_Test<int64_t>(
      {4, 12},
      {2, 3},
      {2, 9},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().sub(args);
      });

  ConstExprTest_Elementwise_Test<float_t>(
      {4.1, 101},
      {0.1, 50.5},
      {4.0, 50.5},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().sub(args);
      });

  ConstExprTest_Elementwise_Test<int64_t>(
      {4, 12},
      {2, 3},
      {8, 36},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().mul(args);
      });

  ConstExprTest_Elementwise_Test<float16_t>(
      {4., 12},
      {0.5, 1.5},
      {2.0, 18},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiOnnxOpset10().mul(args);
      });

  ConstExprTest_Elementwise_Test<int64_t>(
      {4, -12},
      {-3, 8},
      {1, -4},
      [](std::unique_ptr<popart::Builder> &builder,
         const std::vector<TensorId> &args) -> TensorId {
        return builder->aiGraphcoreOpset1().fmod(args);
      });
}
