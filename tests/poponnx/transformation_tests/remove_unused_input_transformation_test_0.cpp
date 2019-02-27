#define BOOST_TEST_MODULE RemoveUnusedInputTest0

#include <boost/test/unit_test.hpp>
#include <onnx/onnx_pb.h>
#include <poponnx/builder.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/tensordata.hpp>

using namespace poponnx;

// Confirm that when removeUunusedInputs is used,
// the two unused initializer inputs are removed
BOOST_AUTO_TEST_CASE(Transformation_RemoveUnused0) {

  auto test = [](bool removeOn) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    TensorInfo info_d{"FLOAT", std::vector<int64_t>{4, 4, 3, 1}};
    TensorInfo info_w{"FLOAT", std::vector<int64_t>{4, 1, 3, 3}};
    float vals_w[4 * 1 * 3 * 3] = {0};
    ConstVoidData weight_data   = {vals_w, info_w};

    auto w0  = builder->addInitializedInputTensor(weight_data);
    auto w1  = builder->addInitializedInputTensor(weight_data);
    auto w2  = builder->addInitializedInputTensor(weight_data);
    auto in0 = builder->addInputTensor(info_d);
    auto h0  = aiOnnx.add({w1, in0});
    auto out = aiOnnx.relu({h0});
    builder->addOutputTensor(h0);
    std::string proto = builder->getModelProto();
    onnx::ModelProto postProto;
    if (removeOn) {
      GraphTransformer gt(proto);
      gt.removeUnusedInputs();
      postProto = io::getModelFromString(gt.getModelProto());
    } else {
      postProto = io::getModelFromString(proto);
    }

    if (removeOn) {
      BOOST_CHECK(postProto.graph().input_size() == 2);
      BOOST_CHECK(postProto.graph().initializer_size() == 1);
    } else {
      BOOST_CHECK(postProto.graph().input_size() == 4);
      BOOST_CHECK(postProto.graph().initializer_size() == 3);
    }
  };

  test(true);
  test(false);
}
