// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprCastTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <memory>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/half.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/l1.hpp>
#include <popart/sgd.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <math.h>

using namespace popart;

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
  auto aiOnnx        = builder->aiOnnxOpset9();
  auto aiGraphcore   = builder->aiGraphcoreOpset1();
  // The two fixed-point tensors which are added together are Constants
  auto i0Id   = aiOnnx.constant(i0cv, "i0cv");
  auto i1Id   = aiOnnx.constant(i1cv, "i1cv");
  auto dataId = builder->addInputTensor(dataInfo);
  auto i01Id  = aiOnnx.add({i0Id, i1Id});
  auto castId = aiOnnx.cast({i01Id}, "FLOAT");
  auto outId  = aiOnnx.matmul({dataId, castId});
  auto lossId = aiGraphcore.l1loss({outId}, 0.1);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("All");
  auto dataFlow  = DataFlow(1, {{outId, art}});
  auto optimizer = ConstSGD(0.01);

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              lossId,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) The Matmul Op is present,
  auto matmuls = ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul);
  BOOST_CHECK(matmuls.size() == 1);
  BOOST_CHECK(matmuls[0]->input->hasIndex(1));
  auto weights = matmuls[0]->input->tensor(1);

  // 2) The shape of the output is correct
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outshape);

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

template <typename T> std::string getTypeName();

template <> std::string getTypeName<int32_t>() { return "INT32"; }
template <> std::string getTypeName<uint32_t>() { return "UINT32"; }
template <> std::string getTypeName<float>() { return "FLOAT"; }
template <> std::string getTypeName<float16_t>() { return "FLOAT16"; }

template <typename FROM, typename TO> void ConstExprTest_AddCastMatMul_Type() {

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

  std::vector<FROM> i0(K);
  std::iota(i0.begin(), i0.end(), 1);
  std::vector<FROM> i1(N);
  std::iota(i1.begin(), i1.end(), 1);
  // TensorInfo dataInfo{"FLOAT", std::vector<int64_t>{M, K}};
  TensorInfo dataInfo{getTypeName<TO>(), std::vector<int64_t>{M, K}};
  ConstVoidData i0cv = {i0.data(),
                        {getTypeName<FROM>(), std::vector<int64_t>{K, 1}}};
  ConstVoidData i1cv = {i1.data(),
                        {getTypeName<FROM>(), std::vector<int64_t>{1, N}}};
  auto builder       = Builder::create();
  auto aiOnnx        = builder->aiOnnxOpset9();
  auto aiGraphcore   = builder->aiGraphcoreOpset1();
  // The two fixed-point tensors which are added together are Constants
  auto i0Id   = aiOnnx.constant(i0cv, "i0cv");
  auto i1Id   = aiOnnx.constant(i1cv, "i1cv");
  auto dataId = builder->addInputTensor(dataInfo);
  auto i01Id  = aiOnnx.add({i0Id, i1Id});
  auto castId = aiOnnx.cast(
      {i01Id}, dataInfo.getDataTypeInfo()->name()); // DataType::FLOAT);
  auto outId  = aiOnnx.matmul({dataId, castId});
  auto lossId = aiGraphcore.l1loss({outId}, 0.1);

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
              Patterns::create({"PostNRepl"}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) The Matmul Op is present,
  auto matmuls = ir.opsOfType(Onnx::AiOnnx::OpSet9::MatMul);
  BOOST_CHECK(matmuls.size() == 1);
  BOOST_CHECK(matmuls[0]->input->hasIndex(1));
  auto weights = matmuls[0]->input->tensor(1);

  // 2) The shape of the output is correct
  BOOST_CHECK(ir.getMainGraphTensors().get(outId)->info.shape() == outshape);

  // 3) The weights inputs to the matmul are correct,
  BOOST_CHECK(weights->info.shape() == weights_shape);
  BOOST_CHECK(weights->tensorType() == TensorType::Const);
  BOOST_CHECK(weights->hasTensorData() == true);
  auto tensorWData = weights->tensorData()->data();
  auto floatWData  = reinterpret_cast<TO *>(tensorWData);
  // The weights were formed as the outer product of two
  // vectors, {1...K} and {1...N}. We therefore expect the very
  // last element of the weights tensor to be K + N
  logging::info("{} ? {}", floatWData[K * N - 1], static_cast<TO>(K + N));
  BOOST_CHECK(floatWData[K * N - 1] == static_cast<TO>(K + N));
}

BOOST_AUTO_TEST_CASE(ConstExprTest_AddCastMatMul_Types) {
  ConstExprTest_AddCastMatMul_Type<int32_t, float_t>();
  ConstExprTest_AddCastMatMul_Type<int32_t, float16_t>();
  ConstExprTest_AddCastMatMul_Type<uint32_t, float_t>();
  ConstExprTest_AddCastMatMul_Type<uint32_t, float16_t>();
}
