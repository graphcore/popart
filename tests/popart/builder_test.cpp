// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE BuilderTest

#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/error.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>

using namespace popart;

bool invalidOpsetId(const error &ex) {
  BOOST_CHECK_EQUAL(
      ex.what(),
      std::string("Invalid opset 6 used to add an operation. Opset for domain "
                  "ai.onnx already defined as 9"));
  return true;
}

BOOST_AUTO_TEST_CASE(Builder_MultiOpset) {

  // Build a model using two opset for the same domain, this will be invalid

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx9 = builder->aiOnnxOpset9();
  auto aiOnnx6 = builder->aiOnnxOpset6();

  TensorInfo shape0{"FLOAT", std::vector<int64_t>{3}};
  auto in0 = builder->addInputTensor(shape0);
  auto in1 = builder->addInputTensor(shape0);
  auto t1  = aiOnnx9.concat({in0, in1}, 0);

  auto in2 = builder->addInputTensor(shape0);

  BOOST_CHECK_EXCEPTION(aiOnnx6.concat({in2, t1}, 0), error, invalidOpsetId);
}
