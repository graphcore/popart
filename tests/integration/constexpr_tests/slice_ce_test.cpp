// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ConstExprSliceTest

#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
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
  auto aiOnnx     = builder->aiOnnxOpset9();
  auto const_node = aiOnnx.constant(const_data, "const_data");
  auto slice_node = aiOnnx.slice({const_node}, {2, 3}, {1, 0}, {0, 1});
  auto in_id      = builder->addInputTensor(in_info);
  auto out_id     = aiOnnx.add({slice_node, in_id});
  auto l1         = builder->aiGraphcoreOpset1().l1loss({out_id}, 0.1);

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("All");
  auto data_flow = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              l1,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the Slice Op is not present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
  // 3) that the shape of the output tensor is as specified.
  Shape ref_shape{1, 3};
  BOOST_CHECK(ir.getMainGraphTensors().get(out_id)->info.shape() == ref_shape);
}

template <typename T> std::string getTypeName();

template <> std::string getTypeName<int32_t>() { return "INT32"; }
template <> std::string getTypeName<float_t>() { return "FLOAT"; }
template <> std::string getTypeName<float16_t>() { return "FLOAT16"; }

template <typename T> void ConstExprTest_Slice0_Type() {
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

  std::vector<T> raw_const_data(7);
  std::iota(raw_const_data.begin(), raw_const_data.end(), 1);
  ConstVoidData const_data = {raw_const_data.data(),
                              {getTypeName<T>(), Shape{2, 4}}};

  TensorInfo in_info{getTypeName<T>(), Shape{1}};

  auto builder    = Builder::create();
  auto aiOnnx     = builder->aiOnnxOpset9();
  auto const_node = aiOnnx.constant(const_data, "const_data");
  auto slice_node = aiOnnx.slice({const_node}, {2, 3}, {1, 0}, {0, 1});
  auto in_id      = builder->addInputTensor(in_info);
  auto out_id     = aiOnnx.add({slice_node, in_id});
  auto l1         = builder->aiGraphcoreOpset1().l1loss({out_id}, 0.1);

  auto proto       = builder->getModelProto();
  auto model_proto = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto art       = AnchorReturnType("All");
  auto data_flow = DataFlow(1, {{out_id, art}});
  auto optimizer = ConstSGD(0.01);
  auto device    = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto,
              InputShapeInfo(),
              data_flow,
              l1,
              &optimizer,
              *device,
              {}, // no SessionOptions
              Patterns({}).enableRuntimeAsserts(false)});

  // Check the ir
  // 1) that the Add Op is present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Add).size() == 1);
  // 2) that the Slice Op is not present,
  BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
  // 3) that the shape of the output tensor is as specified.
  Shape ref_shape{1, 3};
  BOOST_CHECK(ir.getMainGraphTensors().get(out_id)->info.shape() == ref_shape);
}

BOOST_AUTO_TEST_CASE(ConstExprTest_Slice0_Types) {
  ConstExprTest_Slice0_Type<float_t>();
  ConstExprTest_Slice0_Type<float16_t>();
  ConstExprTest_Slice0_Type<int32_t>();
}
