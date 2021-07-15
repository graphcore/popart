// Copyright (c) 2019 Graphcore Ltd. All rights reserved.  #define
// BOOST_TEST_MODULE Basic0TopkTest
#define BOOST_TEST_MODULE ConstOpTest

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/voiddata.hpp>

BOOST_AUTO_TEST_CASE(ConstOp_Opset9) {

  popart::TensorInfo const_shape{"FLOAT", std::vector<int64_t>{4}};
  float const_data[4] = {3.0, 0.14, 0.0015, 0.000092};
  popart::ConstVoidData const_cvdata{const_data, const_shape};

  auto builder    = popart::Builder::create();
  auto aiOnnx     = builder->aiOnnxOpset9();
  auto constId    = aiOnnx.constant(const_cvdata);
  auto identityId = aiOnnx.identity({constId});
  builder->addOutputTensor(identityId);

  // Running `Builder::createFromOnnxModel` runs the onnx model checker.
  // If the onnx model checker detects a problem, it will throw an error and the
  // test will fail.
  auto proto = builder->getModelProto();
  popart::Builder::createFromOnnxModel(proto);
}

// To be enabled when opset 11 is enabled in the builder.
BOOST_AUTO_TEST_CASE(ConstOp_Opset11) {

  popart::TensorInfo const_shape{"FLOAT", std::vector<int64_t>{4}};
  float const_data[4] = {3., 0.14, 0.0015, 0.000092};
  popart::ConstVoidData const_cvdata{const_data, const_shape};

  auto builder    = popart::Builder::create();
  auto aiOnnx     = builder->aiOnnxOpset11();
  auto constId    = aiOnnx.constant(const_cvdata, false);
  auto identityId = aiOnnx.identity({constId});
  builder->addOutputTensor(identityId);

  // Running `Builder::createFromOnnxModel` runs the onnx model checker. If the
  // onnx model checker detects a problem, it will throw an error and the test
  // will fail.
  auto proto = builder->getModelProto();
  popart::Builder::createFromOnnxModel(proto);
}
