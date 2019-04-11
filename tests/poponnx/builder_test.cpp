
#define BOOST_TEST_MODULE BuilderTest

#include <boost/test/unit_test.hpp>

#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/inputshapeinfo.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/tensors.hpp>

using namespace poponnx;

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
